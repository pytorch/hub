# Jekyll::Paginate V2::AutoPages

**AutoPages** are an optional pagination addon that can automatically generate paginated pages for all your collections, tags and categories used in the pages on your site. This is useful if you have a large site where paginating the contents of your collections, tags or category lists provides a better user experience.

<p align="center">
  <img src="https://raw.githubusercontent.com/sverrirs/jekyll-paginate-v2/master/res/autopages-logo.png" height="128" />
</p>


> This feature is based on [code](https://github.com/stevecrozz/lithostech.com/blob/master/_plugins/tag_indexes.rb) written and graciously donated by [Stephen Crosby](https://github.com/stevecrozz). Thanks! :)

* [Site configuration](#site-configuration)
* [Simple configuration](#simple-configuration)
  + [Obtaining the original Tag or Category name](#obtaining-the-original-tag-or-category-name)
* [Advanced configuration](#advanced-configuration)
* [Specialised pages](#specialised-pages)
* [Considerations](#considerations)
  + [Title should not contain pagination macros](#title-should-not-contain-pagination-macros)
* [Example Sites](https://github.com/sverrirs/jekyll-paginate-v2/tree/master/examples)
* [Common issues](#common-issues)

:warning: Please note, this feature is still **experimental** and might not be ready for production yet.

## Site configuration

``` yml
############################################################
# Site configuration for the Auto-Pages feature
# The values here represent the defaults if nothing is set
autopages:

  # Site-wide kill switch, disable here and it doesn't run at all 
  enabled: false

  # Category pages, omit entire config element to disable
  categories: 
    # Optional, the list of layouts that should be processed for every category found in the site
    layouts: 
      - 'autopage_category.html'
    # Optional, the title that each category paginate page should get (:cat is replaced by the Category name)
    title: 'Posts in category :cat'
    # Optional, the permalink for the  pagination page (:cat is replaced), 
    # the pagination permalink path is then appended to this permalink structure
    permalink: '/category/:cat'

  # Collection pages, omit to disable
  collections:
    layouts: 
      - 'autopage_collection.html'
    title: 'Posts in collection :coll' # :coll is replaced by the collection name
    permalink: '/collection/:coll'
  
  # Tag pages, omit to disable
  tags:
    layouts: 
      - 'autopage_tags.html'
    title: 'Posts tagged with :tag' # :tag is replaced by the tag name
    permalink: '/tag/:tag'
```

## Simple configuration
The only thing needed to enable the auto pages is to add the configuration element to the site.config file and then create a layout that the pages should use.

Example of a simple autopage layout can be seen in Example 3 [examples/03-tags/_layouts/autopage_tags.html](https://github.com/sverrirs/jekyll-paginate-v2/blob/master/examples/03-tags/_layouts/autopage_tags.html).

The layout does not need anything special beyond the normal pagination logic (e.g. `for post in paginator.posts` and then the next/prev arrows). You can either name the layouts by the default names (see site configuration section above) or give them a custom name and add them to the `layouts:` configuration for the relevant type of autopage.

### Obtaining the original Tag or Category name
Internally the autopage system will trim and downcase the indexing key (tag, category or collection name). To retrieve the original name of the index key you can use the `autopages.display_name` liquid variable.

``` html
<h1 class="page-heading">All pages tagged with <b>{% if page.autopages %}{{page.autopages.display_name}}{% endif %}</b></h1>
```

This variable returns the untouched key value. As an example if your site uses the tag `Science-Fiction` then

```
page.tag = "science-fiction"
page.autopages.display_name = "Science-Fiction"
```

See [#6](https://github.com/sverrirs/jekyll-paginate-v2/issues/6) for more information.


## Advanced configuration
You can customize the look an feel of the permalink structure and the title for the auto-pages. You can also add front-matter `pagination` configuration to the layout pages you're using to specify things like sorting, filtering and all the other configuration options that are available in the normal pagination generator.

> Keep in mind that when the autopages are paginated the pagination permalink structure and pagination title suffix is appended to them.

## Specialised pages

Special autopages can be used that leverage the powerful filtering and sorting features of the pagination logic. For example you might want to create special pages autopages that paginate all tags in a certain collection or certain tags within a category.

To achieve this you specify the `pagination` front-matter configuration in the autopage layout file.

An example of this can be found in [examples/03-tags/_layouts/autopage_collections_tags.html](https://github.com/sverrirs/jekyll-paginate-v2/blob/master/examples/03-tags/_layouts/autopage_collections_tags.html). This page creates paginated pages for all tags that are found in the special _all_ collections.

## Considerations

### Title should not contain pagination macros
There is no need to include the pagination title macros `:num`, `:max` or `:title` in the title configuration. The autopages will use the title configuration from the pagination configuration itself.

## Common issues
_None reported so far_