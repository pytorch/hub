module Jekyll
  module PaginateV2::AutoPages

    # The default configuration for the AutoPages
    DEFAULT = {
      'enabled'     => false,
      'tags'        => {
        'layouts'       => ['autopage_tags.html'],
        'title'         => 'Posts tagged with :tag',
        'permalink'     => '/tag/:tag',
        'enabled'       => true,
        'slugify'       => {
                              'mode' => 'none', # [raw default pretty ascii latin], none gives back the same string
                              'cased'=> false # If cased is true, all uppercase letters in the result string are replaced with their lowercase counterparts.
                            }
      },
      'categories'  => {
        'layouts'       => ['autopage_category.html'],
        'title'         => 'Posts in category :cat',
        'permalink'     => '/category/:cat',
        'enabled'       => true,
        'slugify'       => {
                              'mode' => 'none', # [raw default pretty ascii latin], none gives back the same string
                              'cased'=> false # If cased is true, all uppercase letters in the result string are replaced with their lowercase counterparts.
                            }
      },
      'collections' => {
        'layouts'       => ['autopage_collection.html'],
        'title'         => 'Posts in collection :coll',
        'permalink'     => '/collection/:coll',
        'enabled'       => true,
        'slugify'       => {
                              'mode' => 'none', # [raw default pretty ascii latin], none gives back the same string
                              'cased'=> false # If cased is true, all uppercase letters in the result string are replaced with their lowercase counterparts.
                            }
      } 
    }

  end # module PaginateV2::AutoPages
end # module Jekyll
