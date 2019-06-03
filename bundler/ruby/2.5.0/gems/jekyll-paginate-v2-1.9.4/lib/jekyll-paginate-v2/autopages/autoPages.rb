module Jekyll
  module PaginateV2::AutoPages

    #
    # This function is called right after the main generator is triggered by Jekyll
    # This code is adapted from Stephen Crosby's code https://github.com/stevecrozz
    def self.create_autopages(site)

      # Get the configuration for the auto pages
      autopage_config = Jekyll::Utils.deep_merge_hashes(DEFAULT, site.config['autopages'] || {})
      pagination_config = Jekyll::Utils.deep_merge_hashes(Jekyll::PaginateV2::Generator::DEFAULT, site.config['pagination'] || {})

      # If disabled then don't do anything
      if !autopage_config['enabled'] || autopage_config['enabled'].nil?
        Jekyll.logger.info "AutoPages:","Disabled/Not configured in site.config."
        return
      end

      # TODO: Should I detect here and disable if we're running the legacy paginate code???!

      # Simply gather all documents across all pages/posts/collections that we have
      # we could be generating quite a few empty pages but the logic is just vastly simpler than trying to 
      # figure out what tag/category belong to which collection.
      posts_to_use = Utils.collect_all_docs(site.collections)

      ###############################################
      # Generate the Tag pages if enabled
      createtagpage_lambda = lambda do | autopage_tag_config, pagination_config, layout_name, tag, tag_original_name |
        site.pages << TagAutoPage.new(site, site.dest, autopage_tag_config, pagination_config, layout_name, tag, tag_original_name)
      end
      autopage_create(autopage_config, pagination_config, posts_to_use, 'tags', 'tags', createtagpage_lambda) # Call the actual function
      

      ###############################################
      # Generate the category pages if enabled
      createcatpage_lambda = lambda do | autopage_cat_config, pagination_config, layout_name, category, category_original_name |
        site.pages << CategoryAutoPage.new(site, site.dest, autopage_cat_config, pagination_config, layout_name, category, category_original_name)
      end
      autopage_create(autopage_config, pagination_config,posts_to_use, 'categories', 'categories', createcatpage_lambda) # Call the actual function
      
      ###############################################
      # Generate the Collection pages if enabled
      createcolpage_lambda = lambda do | autopage_col_config, pagination_config, layout_name, coll_name, coll_original_name |
        site.pages << CollectionAutoPage.new(site, site.dest, autopage_col_config, pagination_config, layout_name, coll_name, coll_original_name)
      end
      autopage_create(autopage_config, pagination_config,posts_to_use, 'collections', '__coll', createcolpage_lambda) # Call the actual function
    
    end # create_autopages


    # STATIC: this function actually performs the steps to generate the autopages. It uses a lambda function to delegate the creation of the individual
    #         page types to the calling code (this way all features can reuse the logic).
    #
    def self.autopage_create(autopage_config, pagination_config, posts_to_use, configkey_name, indexkey_name, createpage_lambda )
      if !autopage_config[configkey_name].nil?
        ap_sub_config = autopage_config[configkey_name]
        if ap_sub_config ['enabled']
          Jekyll.logger.info "AutoPages:","Generating #{configkey_name} pages"

          # Roll through all documents in the posts collection and extract the tags
          index_keys = Utils.ap_index_posts_by(posts_to_use, indexkey_name) # Cannot use just the posts here, must use all things.. posts, collections...

          index_keys.each do |index_key, value|
            # Iterate over each layout specified in the config
            ap_sub_config ['layouts'].each do | layout_name |
              # Use site.dest here as these pages are never created in the actual source but only inside the _site folder
              createpage_lambda.call(ap_sub_config, pagination_config, layout_name, index_key, value[-1]) # the last item in the value array will be the display name
            end
          end
        else
          Jekyll.logger.info "AutoPages:","#{configkey_name} pages are disabled/not configured in site.config."
        end
      end
    end

  end # module PaginateV2
end # module Jekyll
