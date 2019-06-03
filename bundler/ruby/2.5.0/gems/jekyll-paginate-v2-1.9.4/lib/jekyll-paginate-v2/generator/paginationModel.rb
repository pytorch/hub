module Jekyll
  module PaginateV2::Generator

    #
    # The main model for the pagination, handles the orchestration of the pagination and calling all the necessary bits and bobs needed :)
    #
    class PaginationModel

      @debug = false # is debug output enabled?
      @logging_lambda = nil # The lambda to use for logging
      @page_add_lambda = nil # The lambda used to create pages and add them to the site
      @page_remove_lambda = nil # Lambda to remove a page from the site.pages collection
      @collection_by_name_lambda = nil # Lambda to get all documents/posts in a particular collection (by name)

      # ctor
      def initialize(logging_lambda, page_add_lambda, page_remove_lambda, collection_by_name_lambda)
        @logging_lambda = logging_lambda
        @page_add_lambda = page_add_lambda
        @page_remove_lambda = page_remove_lambda
        @collection_by_name_lambda = collection_by_name_lambda
      end


      def run(default_config, site_pages, site_title)
        # By default if pagination is enabled we attempt to find all index.html pages in the site
        templates = self.discover_paginate_templates(site_pages)
        if( templates.size.to_i <= 0 )
          @logging_lambda.call "Is enabled, but I couldn't find any pagination page. Skipping pagination. "+
          "Pages must have 'pagination: enabled: true' in their front-matter for pagination to work.", "warn"
          return
        end

        # Now for each template page generate the paginator for it
        templates.each do |template|
          # All pages that should be paginated need to include the pagination config element
          if template.data['pagination'].is_a?(Hash)
            template_config = Jekyll::Utils.deep_merge_hashes(default_config, template.data['pagination'] || {})

            # Handling deprecation of configuration values
            self._fix_deprecated_config_features(template_config)

            @debug = template_config['debug'] # Is debugging enabled on the page level

            self._debug_print_config_info(template_config, template.path)
            
            # Only paginate the template if it is explicitly enabled
            # requiring this makes the logic simpler as I don't need to determine which index pages 
            # were generated automatically and which weren't
            if( template_config['enabled'] )
              if !@debug
                @logging_lambda.call "found page: "+template.path, 'debug'
              end

              # Request all documents in all collections that the user has requested 
              all_posts = self.get_docs_in_collections(template_config['collection'])

              # Create the necessary indexes for the posts
              all_categories = PaginationIndexer.index_posts_by(all_posts, 'categories')
              all_categories['posts'] = all_posts; # Populate a category for all posts (this is here for backward compatibility, do not use this as it will be decommissioned 2018-01-01) 
                                                  # (this is a default and must not be used in the category system)
              all_tags = PaginationIndexer.index_posts_by(all_posts, 'tags')
              all_locales = PaginationIndexer.index_posts_by(all_posts, 'locale')

              # TODO: NOTE!!! This whole request for posts and indexing results could be cached to improve performance, leaving like this for now during testing

              # Now construct the pagination data for this template page
              self.paginate(template, template_config, site_title, all_posts, all_tags, all_categories, all_locales)
            end
          end
        end #for

        # Return the total number of templates found
        return templates.size.to_i
      end # function run

      #
      # This function is here to retain the old compatability logic with the jekyll-paginate gem
      # no changes should be made to this function and it should be retired and deleted after 2018-01-01
      # (REMOVE AFTER 2018-01-01)
      #
      def run_compatability(legacy_config, site_pages, site_title, all_posts)

        # Decomissioning error
        if( date = Date.strptime("20180101","%Y%m%d") <= Date.today )
          raise ArgumentError.new("Legacy jekyll-paginate configuration compatibility mode has expired. Please upgrade to jekyll-paginate-v2 configuration.")
        end

        # Two month warning or general notification
        if( date = Date.strptime("20171101","%Y%m%d") <= Date.today )
          @logging_lambda.call "Legacy pagination logic will stop working on Jan 1st 2018, update your configs before that time.", "warn"
        else
          @logging_lambda.call "Detected legacy jekyll-paginate logic running. "+
          "Please update your configs to use the jekyll-paginate-v2 logic. This compatibility function "+
          "will stop working after Jan 1st 2018 and your site build will throw an error.", "warn"
        end

        if template = CompatibilityUtils.template_page(site_pages, legacy_config['legacy_source'], legacy_config['permalink'])
          CompatibilityUtils.paginate(legacy_config, all_posts, template, @page_add_lambda)
        else
          @logging_lambda.call "Legacy pagination is enabled, but I couldn't find " +
          "an index.html page to use as the pagination page. Skipping pagination.", "warn"
        end
      end # function run_compatability (REMOVE AFTER 2018-01-01)

      # Returns the combination of all documents in the collections that are specified
      # raw_collection_names can either be a list of collections separated by a ',' or ' ' or a single string
      def get_docs_in_collections(raw_collection_names)
        if raw_collection_names.is_a?(String)
          collection_names = raw_collection_names.split(/;|,|\s/)
        else
          collection_names = raw_collection_names
        end

        docs = []
        # Now for each of the collections get the docs
        collection_names.each do |coll_name|
          # Request all the documents for the collection in question, and join it with the total collection 
          docs += @collection_by_name_lambda.call(coll_name.downcase.strip)
        end

        # Hidden documents should not not be processed anywhere.
        docs = docs.reject { |doc| doc['hidden'] }

        return docs
      end

      def _fix_deprecated_config_features(config)
        keys_to_delete = []

        # As of v1.5.1 the title_suffix is deprecated and 'title' should be used 
        # but only if title has not been defined already!
        if( !config['title_suffix'].nil? )
          if( config['title'].nil? )
            config['title'] = ":title" + config['title_suffix'].to_s # Migrate the old key to title
          end
          keys_to_delete << "title_suffix" # Always remove the depricated key if found
        end

        # Delete the depricated keys
        config.delete_if{ |k,| keys_to_delete.include? k }
      end

      def _debug_print_config_info(config, page_path)
        r = 20
        f = "Pagination: ".rjust(20)
        # Debug print the config
        if @debug
          puts f + "----------------------------"
          puts f + "Page: "+page_path.to_s
          puts f + " Active configuration"
          puts f + "  Enabled: ".ljust(r) + config['enabled'].to_s
          puts f + "  Items per page: ".ljust(r) + config['per_page'].to_s
          puts f + "  Permalink: ".ljust(r) + config['permalink'].to_s
          puts f + "  Title: ".ljust(r) + config['title'].to_s
          puts f + "  Limit: ".ljust(r) + config['limit'].to_s
          puts f + "  Sort by: ".ljust(r) + config['sort_field'].to_s
          puts f + "  Sort reverse: ".ljust(r) + config['sort_reverse'].to_s
          
          puts f + " Active Filters"
          puts f + "  Collection: ".ljust(r) + config['collection'].to_s
          puts f + "  Category: ".ljust(r) + (config['category'].nil? || config['category'] == "posts" ? "[Not set]" : config['category'].to_s)
          puts f + "  Tag: ".ljust(r) + (config['tag'].nil? ? "[Not set]" : config['tag'].to_s)
          puts f + "  Locale: ".ljust(r) + (config['locale'].nil? ? "[Not set]" : config['locale'].to_s)

          if config['legacy'] 
            puts f + " Legacy Paginate Code Enabled"
            puts f + "  Legacy Paginate: ".ljust(r) + config['per_page'].to_s
            puts f + "  Legacy Source: ".ljust(r) + config['legacy_source'].to_s
            puts f + "  Legacy Path: ".ljust(r) + config['paginate_path'].to_s
          end
        end
      end

      def _debug_print_filtering_info(filter_name, before_count, after_count)
        # Debug print the config
        if @debug
          puts "Pagination: ".rjust(20) + " Filtering by: "+filter_name.to_s.ljust(9) + " " + before_count.to_s.rjust(3) + " => " + after_count.to_s  
        end
      end
      
      #
      # Rolls through all the pages passed in and finds all pages that have pagination enabled on them.
      # These pages will be used as templates
      #
      # site_pages - All pages in the site
      #
      def discover_paginate_templates(site_pages)
        candidates = []
        site_pages.select do |page|
          # If the page has the enabled config set, supports any type of file name html or md
          if page.data['pagination'].is_a?(Hash) && page.data['pagination']['enabled']
            candidates << page
          end
        end
        return candidates
      end # function discover_paginate_templates
            
      # Paginates the blog's posts. Renders the index.html file into paginated
      # directories, e.g.: page2/index.html, page3/index.html, etc and adds more
      # site-wide data.
      #
      # site - The Site.
      # template - The index.html Page that requires pagination.
      # config - The configuration settings that should be used
      #
      def paginate(template, config, site_title, all_posts, all_tags, all_categories, all_locales)
        # By default paginate on all posts in the site
        using_posts = all_posts
        
        # Now start filtering out any posts that the user doesn't want included in the pagination
        before = using_posts.size.to_i
        using_posts = PaginationIndexer.read_config_value_and_filter_posts(config, 'category', using_posts, all_categories)
        self._debug_print_filtering_info('Category', before, using_posts.size.to_i)
        before = using_posts.size.to_i
        using_posts = PaginationIndexer.read_config_value_and_filter_posts(config, 'tag', using_posts, all_tags)
        self._debug_print_filtering_info('Tag', before, using_posts.size.to_i)
        before = using_posts.size.to_i
        using_posts = PaginationIndexer.read_config_value_and_filter_posts(config, 'locale', using_posts, all_locales)
        self._debug_print_filtering_info('Locale', before, using_posts.size.to_i)
        
        # Apply sorting to the posts if configured, any field for the post is available for sorting
        if config['sort_field']
          sort_field = config['sort_field'].to_s

          # There is an issue in Jekyll related to lazy initialized member variables that causes iterators to 
          # break when accessing an uninitialized value during iteration. This happens for document.rb when the <=> compaison function 
          # is called (as this function calls the 'date' field which for drafts are not initialized.)
          # So to unblock this common issue for the date field I simply iterate once over every document and initialize the .date field explicitly
          if @debug
            puts "Pagination: ".rjust(20) + "Rolling through the date fields for all documents"
          end
          using_posts.each do |u_post|
            if u_post.respond_to?('date')
              tmp_date = u_post.date
              if( !tmp_date || tmp_date.nil? )
                if @debug
                  puts "Pagination: ".rjust(20) + "Explicitly assigning date for doc: #{u_post.data['title']} | #{u_post.path}"
                end
                u_post.date = File.mtime(u_post.path)
              end
            end
          end

          using_posts.sort!{ |a,b| Utils.sort_values(Utils.sort_get_post_data(a.data, sort_field), Utils.sort_get_post_data(b.data, sort_field)) }

          if config['sort_reverse']
            using_posts.reverse!
          end
        end
               
        # Calculate the max number of pagination-pages based on the configured per page value
        total_pages = Utils.calculate_number_of_pages(using_posts, config['per_page'])
        
        # If a upper limit is set on the number of total pagination pages then impose that now
        if config['limit'] && config['limit'].to_i > 0 && config['limit'].to_i < total_pages
          total_pages = config['limit'].to_i
        end

        #### BEFORE STARTING REMOVE THE TEMPLATE PAGE FROM THE SITE LIST!
        @page_remove_lambda.call( template )
        
        # list of all newly created pages
        newpages = []

        # Consider the default index page name and extension
        indexPageName = config['indexpage'].nil? ? '' : config['indexpage'].split('.')[0]
        indexPageExt =  config['extension'].nil? ? '' : Utils.ensure_leading_dot(config['extension'])
        indexPageWithExt = indexPageName + indexPageExt

        # In case there are no (visible) posts, generate the index file anyway
        total_pages = 1 if total_pages.zero?

        # Now for each pagination page create it and configure the ranges for the collection
        # This .pager member is a built in thing in Jekyll and defines the paginator implementation
        # Simpy override to use mine
        (1..total_pages).each do |cur_page_nr|

          # 1. Create the in-memory page
          #    External Proc call to create the actual page for us (this is passed in when the pagination is run)
          newpage = PaginationPage.new( template, cur_page_nr, total_pages, indexPageWithExt )

          # 2. Create the url for the in-memory page (calc permalink etc), construct the title, set all page.data values needed
          paginated_page_url = config['permalink']
          first_index_page_url = ""
          if template.data['permalink']
            first_index_page_url = Utils.ensure_trailing_slash(template.data['permalink'])
          else
            first_index_page_url = Utils.ensure_trailing_slash(template.dir)
          end
          paginated_page_url = File.join(first_index_page_url, paginated_page_url)
          
          # 3. Create the pager logic for this page, pass in the prev and next page numbers, assign pager to in-memory page
          newpage.pager = Paginator.new( config['per_page'], first_index_page_url, paginated_page_url, using_posts, cur_page_nr, total_pages, indexPageName, indexPageExt)

          # Create the url for the new page, make sure we prepend any permalinks that are defined in the template page before
          if newpage.pager.page_path.end_with? '/'
            newpage.set_url(File.join(newpage.pager.page_path, indexPageWithExt))
          elsif newpage.pager.page_path.end_with? indexPageExt
            # Support for direct .html files
            newpage.set_url(newpage.pager.page_path)
          else
            # Support for extensionless permalinks
            newpage.set_url(newpage.pager.page_path+indexPageExt)
          end

          if( template.data['permalink'] )
            newpage.data['permalink'] = newpage.pager.page_path
          end

          # Transfer the title across to the new page
          if( !template.data['title'] )
            tmp_title = site_title
          else
            tmp_title = template.data['title']
          end
          # If the user specified a title suffix to be added then let's add that to all the pages except the first
          if( cur_page_nr > 1 && config.has_key?('title') )
            newpage.data['title'] = "#{Utils.format_page_title(config['title'], tmp_title, cur_page_nr, total_pages)}"
          else
            newpage.data['title'] = tmp_title
          end

          # Signals that this page is automatically generated by the pagination logic
          # (we don't do this for the first page as it is there to mask the one we removed)
          if cur_page_nr > 1
            newpage.data['autogen'] = "jekyll-paginate-v2"
          end
          
          # Add the page to the site
          @page_add_lambda.call( newpage )

          # Store the page in an internal list for later referencing if we need to generate a pagination number path later on
          newpages << newpage
        end #each.do total_pages

        # Now generate the pagination number path, e.g. so that the users can have a prev 1 2 3 4 5 next structure on their page
        # simplest is to include all of the links to the pages preceeding the current one
        # (e.g for page 1 you get the list 2, 3, 4.... and for page 2 you get the list 3,4,5...)
        if( config['trail'] && !config['trail'].nil? && newpages.size.to_i > 1 )
          trail_before = [config['trail']['before'].to_i, 0].max
          trail_after = [config['trail']['after'].to_i, 0].max
          trail_length = trail_before + trail_after + 1

          if( trail_before > 0 || trail_after > 0 )
            newpages.select do | npage |
              idx_start = [ npage.pager.page - trail_before - 1, 0].max # Selecting the beginning of the trail
              idx_end = [idx_start + trail_length, newpages.size.to_i].min # Selecting the end of the trail

              # Always attempt to maintain the max total of <trail_length> pages in the trail (it will look better if the trail doesn't shrink)
              if( idx_end - idx_start < trail_length )
                # Attempt to pad the beginning if we have enough pages
                idx_start = [idx_start - ( trail_length - (idx_end - idx_start) ), 0].max # Never go beyond the zero index
              end              

              # Convert the newpages array into a two dimensional array that has [index, page_url] as items
              #puts( "Trail created for page #{npage.pager.page} (idx_start:#{idx_start} idx_end:#{idx_end})")
              npage.pager.page_trail = newpages[idx_start...idx_end].each_with_index.map {|ipage,idx| PageTrail.new(idx_start+idx+1, ipage.pager.page_path, ipage.data['title'])}
              #puts( npage.pager.page_trail )
            end #newpages.select
          end #if trail_before / trail_after
        end # if config['trail']

      end # function paginate

    end # class PaginationV2

  end # module PaginateV2
end # module Jekyll
