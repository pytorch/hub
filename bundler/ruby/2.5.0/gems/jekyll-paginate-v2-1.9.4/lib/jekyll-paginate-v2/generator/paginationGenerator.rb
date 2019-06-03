module Jekyll
  module PaginateV2::Generator
  
    #
    # The main entry point into the generator, called by Jekyll
    # this function extracts all the necessary information from the jekyll end and passes it into the pagination 
    # logic. Additionally it also contains all site specific actions that the pagination logic needs access to
    # (such as how to create new pages)
    # 
    class PaginationGenerator < Generator
      # This generator is safe from arbitrary code execution.
      safe true

      # This generator should be passive with regard to its execution
      priority :lowest
      
      # Generate paginated pages if necessary (Default entry point)
      # site - The Site.
      #
      # Returns nothing.
      def generate(site)
      #begin
        # Generate the AutoPages first
        PaginateV2::AutoPages.create_autopages(site)

        # Retrieve and merge the pagination configuration from the site yml file
        default_config = Jekyll::Utils.deep_merge_hashes(DEFAULT, site.config['pagination'] || {})

        # Compatibility Note: (REMOVE AFTER 2018-01-01)
        # If the legacy paginate logic is configured then read those values and merge with config
        if !site.config['paginate'].nil?
          Jekyll.logger.info "Pagination:","Legacy paginate configuration settings detected and will be used."
          # You cannot run both the new code and the old code side by side
          if !site.config['pagination'].nil?
            err_msg = "The new jekyll-paginate-v2 and the old jekyll-paginate logic cannot both be configured in the site config at the same time. Please disable the old 'paginate:' config settings by either omitting the values or setting them to 'paginate:off'."
            Jekyll.logger.error err_msg 
            raise ArgumentError.new(err_msg)
          end

          default_config['per_page'] = site.config['paginate'].to_i
          default_config['legacy_source'] = site.config['source']
          if !site.config['paginate_path'].nil?
            default_config['permalink'] = site.config['paginate_path'].to_s
          end
          # In case of legacy, enable pagination by default
          default_config['enabled'] = true
          default_config['legacy'] = true
        end # Compatibility END (REMOVE AFTER 2018-01-01)

        # If disabled then simply quit
        if !default_config['enabled']
          Jekyll.logger.info "Pagination:","Disabled in site.config."
          return
        end

        # Handle deprecation of settings and features
        if( !default_config['title_suffix' ].nil? )
          Jekyll::Deprecator.deprecation_message "Pagination: The 'title_suffix' configuration has been deprecated. Please use 'title'. See https://github.com/sverrirs/jekyll-paginate-v2/blob/master/README-GENERATOR.md#site-configuration"
        end

        Jekyll.logger.debug "Pagination:","Starting"

        ################ 0 #################### 
        # Get all pages in the site (this will be used to find the pagination templates)
        all_pages = site.pages

        # Get the default title of the site (used as backup when there is no title available for pagination)
        site_title = site.config['title']

        ################ 1 #################### 
        # Specify the callback function that returns the correct docs/posts based on the collection name
        # "posts" are just another collection in Jekyll but a specialized version that require timestamps
        # This collection is the default and if the user doesn't specify a collection in their front-matter then that is the one we load
        # If the collection is not found then empty array is returned
        collection_by_name_lambda = lambda do |collection_name|
          coll = []
          if collection_name == "all"
            # the 'all' collection_name is a special case and includes all collections in the site (except posts!!)
            # this is useful when you want to list items across multiple collections
            site.collections.each do |coll_name, coll_data|
              if( !coll_data.nil? && coll_name != 'posts')
                coll += coll_data.docs.select { |doc| !doc.data.has_key?('pagination') } # Exclude all pagination pages
              end
            end
          else
            # Just the one collection requested
            if !site.collections.has_key?(collection_name)
              return []
            end

            coll = site.collections[collection_name].docs.select { |doc| !doc.data.has_key?('pagination') } # Exclude all pagination pages
          end
          return coll
        end

        ################ 2 ####################
        # Create the proc that constructs the real-life site page
        # This is necessary to decouple the code from the Jekyll site object
        page_add_lambda = lambda do | newpage |
          site.pages << newpage # Add the page to the site so that it is generated correctly
          return newpage # Return the site to the calling code
        end

        ################ 2.5 ####################
        # lambda that removes a page from the site pages list
        page_remove_lambda = lambda do | page_to_remove |
          site.pages.delete_if {|page| page == page_to_remove } 
        end

        ################ 3 ####################
        # Create a proc that will delegate logging
        # Decoupling Jekyll specific logging
        logging_lambda = lambda do | message, type="info" |
          if type == 'debug'
            Jekyll.logger.debug "Pagination:","#{message}"
          elsif type == 'error'
            Jekyll.logger.error "Pagination:", "#{message}"
          elsif type == 'warn'
            Jekyll.logger.warn "Pagination:", "#{message}"
          else
            Jekyll.logger.info "Pagination:", "#{message}"
          end
        end

        ################ 4 ####################
        # Now create and call the model with the real-life page creation proc and site data
        model = PaginationModel.new(logging_lambda, page_add_lambda, page_remove_lambda, collection_by_name_lambda)
        if( default_config['legacy'] ) #(REMOVE AFTER 2018-01-01)
          Jekyll.logger.warn "Pagination:", "You are running jekyll-paginate backwards compatible pagination logic. Please ignore all earlier warnings displayed related to the old jekyll-paginate gem."
          all_posts = site.site_payload['site']['posts'].reject { |post| post['hidden'] }
          model.run_compatability(default_config, all_pages, site_title, all_posts) #(REMOVE AFTER 2018-01-01)
        else
          count = model.run(default_config, all_pages, site_title)
          Jekyll.logger.info "Pagination:", "Complete, processed #{count} pagination page(s)"
        end

      #rescue => ex
      #  puts ex.backtrace
      #  raise
      #end
      end # function generate
    end # class PaginationGenerator

  end # module PaginateV2
end # module Jekyll
