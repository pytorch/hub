module Jekyll
  module PaginateV2::AutoPages

    class BaseAutoPage < Jekyll::Page
      def initialize(site, base, autopage_config, pagination_config, layout_name, set_autopage_data_lambda, get_autopage_permalink_lambda, get_autopage_title_lambda, display_name)
        @site = site
        @base = base
        @name = 'index.html'
                
        layout_dir = '_layouts'

        # Path is only used by the convertible module and accessed below when calling read_yaml
        # Handling themes stored in a gem
        @path = if site.in_theme_dir(site.source) == site.source # we're in a theme
                  site.in_theme_dir(site.source, layout_dir, layout_name)
                else
                  site.in_source_dir(site.source, layout_dir, layout_name)
                end
        
        self.process(@name) # Creates the base name and extension
        self.read_yaml(File.join(site.source, layout_dir), layout_name)

        # Merge the config with any config that might already be defined in the layout
        pagination_layout_config = Jekyll::Utils.deep_merge_hashes( pagination_config, self.data['pagination'] || {} )

        # Read any possible autopage overrides in the layout page
        autopage_layout_config = Jekyll::Utils.deep_merge_hashes( autopage_config, self.data['autopages'] || {} )
        
        # Now set the page specific pagination data
        set_autopage_data_lambda.call(pagination_layout_config)

        # Get permalink structure
        permalink_formatted = get_autopage_permalink_lambda.call(autopage_layout_config['permalink'])

        # Construct the title
        page_title = autopage_layout_config['title']
               
        # NOTE: Should we set this before calling read_yaml as that function validates the permalink structure 
        self.data['permalink'] = permalink_formatted
        @url = File.join(permalink_formatted, @name)
        @dir = permalink_formatted

        self.data['layout'] = File.basename(layout_name, File.extname(layout_name))
        self.data['title'] = get_autopage_title_lambda.call( page_title )
        self.data['pagination']  = pagination_layout_config # Store the pagination configuration

        # Add the auto page flag in there to be able to detect the page (necessary when figuring out where to load it from)
        # TODO: Need to re-think this variable!!!
        self.data['autopage'] = {"layout_path" => File.join( layout_dir, layout_name ), 'display_name' => display_name.to_s }

        data.default_proc = proc do |_, key|
          site.frontmatter_defaults.find(File.join(layout_dir, layout_name), type, key)
        end

        # Trigger a page event
        #Jekyll::Hooks.trigger :pages, :post_init, self
      end #function initialize
    end #class BaseAutoPage
  end # module PaginateV2
end # module Jekyll