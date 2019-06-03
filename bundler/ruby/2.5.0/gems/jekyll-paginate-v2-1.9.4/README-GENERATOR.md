# Jekyll::Paginate V2::Generator

The **Generator** forms the core of the pagination logic. It is responsible for reading the posts and collections in your site and split them correctly across multiple pages according to the supplied configuration. It also performs the necessary functions to link to the previous and next pages in the page-sets that it generates. 
<p align="center">
  <img src="https://raw.githubusercontent.com/sverrirs/jekyll-paginate-v2/master/res/generator-logo.png" height="128" />
</p>

> The code was based on the original design of [jekyll-paginate](https://github.com/jekyll/jekyll-paginate) and features were sourced from discussions such as [#27](https://github.com/jekyll/jekyll-paginate/issues/27) (thanks [Günter Kits](https://github.com/gynter)).

* [Site configuration](#site-configuration)
* [Page configuration](#page-configuration)
* [Backwards compatibility](#backwards-compatibility-with-jekyll-paginate)
* [Example Sites](https://github.com/sverrirs/jekyll-paginate-v2/tree/master/examples)
* [Paginating collections](#paginating-collections)
  + [Single collection](#paginating-a-single-collection)
  + [Multiple collection](#paginating-multiple-collections)
  + [The special 'all' collection](#the-special-all-collection)
* [How to paginate categories, tags, locales](#paginate-categories-tags-locales)
  + [Filtering categories](#filtering-categories)
  + [Filtering tags](#filtering-tags)
  + [Filtering locales](#filtering-locales)
* [How to paginate on combination of filters](#paginate-on-combination-of-filters)
* [Overriding site configuration](#configuration-overrides)
* [Advanced Sorting](#advanced-sorting)
* [Creating Pagination Trails](#creating-pagination-trails)
* [How to detect auto-generated pages](#detecting-generated-pagination-pages)
* [Formatting page titles](#formatting-page-titles)
* [Reading pagination meta information](#reading-pagination-meta-information)
* [How to generate a JSON API](#generating-a-json-api)
* [Renaming pagination file names](#renaming-pagination-file-names)
* [Common issues](#common-issues)
    - [Dependency Error after installing](#i-keep-getting-a-dependency-error-when-running-jekyll-serve-after-installing-this-gem)
    - [Bundler error upgrading gem (Bundler::GemNotFound)](#im-getting-a-bundler-error-after-upgrading-the-gem-bundlergemnotfound)
    - [Bundler error running gem (Gem::LoadError)](#im-getting-a-bundler-error-after-upgrading-the-gem-gemloaderror)
    - [Pagination pages are not found](#my-pagination-pages-are-not-being-found-couldnt-find-any-pagination-page-skipping-pagination)
    - [Categories cause excess folder nesting](#my-pages-are-being-nested-multiple-levels-deep)
    - [Pagination pages overwriting each others pages](#my-pagination-pages-are-overwriting-each-others-pages)

## Site configuration

The pagination gem is configured in the site's `_config.yml` file by including the `pagination` configuration element

``` yml
############################################################
# Site configuration for the Jekyll 3 Pagination Gem
# The values here represent the defaults if nothing is set
pagination:
  
  # Site-wide kill switch, disabled here it doesn't run at all 
  enabled: true

  # Set to 'true' to enable pagination debugging. This can be enabled in the site config or only for individual pagination pages
  debug: false

  # The default document collection to paginate if nothing is specified ('posts' is default)
  collection: 'posts'

  # How many objects per paginated page, used to be `paginate` (default: 0, means all)
  per_page: 10

  # The permalink structure for the paginated pages (this can be any level deep)
  permalink: '/page/:num/' # Pages are index.html inside this folder (default)
  #permalink: '/page/:num.html' # Pages are simple html files 
  #permalink: '/page/:num' # Pages are html files, linked jekyll extensionless permalink style.

  # Optional the title format for the paginated pages (supports :title for original page title, :num for pagination page number, :max for total number of pages)
  title: ':title - page :num'

  # Limit how many pagenated pages to create (default: 0, means all)
  limit: 0
  
  # Optional, defines the field that the posts should be sorted on (omit to default to 'date')
  sort_field: 'date'

  # Optional, sorts the posts in reverse order (omit to default decending or sort_reverse: true)
  sort_reverse: true

  # Optional, the default category to use, omit or just leave this as 'posts' to get a backwards-compatible behavior (all posts)
  category: 'posts'

  # Optional, the default tag to use, omit to disable
  tag: ''

  # Optional, the default locale to use, omit to disable (depends on a field 'locale' to be specified in the posts, 
  # in reality this can be any value, suggested are the Microsoft locale-codes (e.g. en_US, en_GB) or simply the ISO-639 language code )
  locale: '' 

 # Optional,omit or set both before and after to zero to disable. 
 # Controls how the pagination trail for the paginated pages look like. 
  trail: 
    before: 2
    after: 2

  # Optional, the default file extension for generated pages (e.g html, json, xml).
  # Internally this is set to html by default
  extension: html

  # Optional, the default name of the index file for generated pages (e.g. 'index.html')
  # Without file extension
  indexpage: 'index'

############################################################
```

Also ensure that you remove the old 'jekyll-paginate' gem from your `gems` list and add this new gem instead

``` yml
gems: [jekyll-paginate-v2]
```

## Page configuration

To enable pagination on a page then simply include the minimal pagination configuration in the page front-matter:

``` yml
---
layout: page
pagination: 
  enabled: true
---
```

Then you can use the normal `paginator.posts` logic to iterate through the posts.

``` html
{% for post in paginator.posts %}
  <h1>{{ post.title }}</h1>
{% endfor %}
```

And to display pagination links, simply

``` html
{% if paginator.total_pages > 1 %}
<ul>
  {% if paginator.previous_page %}
  <li>
    <a href="{{ paginator.previous_page_path | prepend: site.baseurl }}">Newer</a>
  </li>
  {% endif %}
  {% if paginator.next_page %}
  <li>
    <a href="{{ paginator.next_page_path | prepend: site.baseurl }}">Older</a>
  </li>
  {% endif %}
</ul>
{% endif %}
```

> All posts that have the `hidden: true` in their front matter are ignored by the pagination logic.

Following fields area available on the `paginator` object

| Field | Description |
| --- | --- |
| per_page | Maximum number of posts or documents on each pagination page. |
| posts | The list of post objects that belong to this pagination page. |
| total_posts | Total number of posts included in pagination. |
| total_pages | Total number of pagination pages created. |
| page | Number of the current pagination page. |
| page_path | The relative Url path of the current pagination page. |
| previous_page | Number of the previous page in the pagination. Nil if no previous page is available. |
| previous_page_path | The relative Url of the previous page. Nil if no previous page is available. |
| next_page | Number of the next page in the pagination. Nil if there is no next page available. |
| next_page_path | The relative Url of the next page in the pagination. Nil if there is no next page available. |
| first_page | Number of the first page in the pagination (usually this is `1`). |
| first_page_path | The relative Url of the first page in the pagination. |
| last_page | Number of the last page in the pagination (this is equal to `total_pages`). |
| last_page_path | The relative Url of the last page in the pagination. |
| page_trail | The [pagination trail](#creating-pagination-trails) structure |


The code is fully backwards compatible and you will have access to all the normal paginator variables defined in the [official jekyll documentation](https://jekyllrb.com/docs/pagination/#liquid-attributes-available). 

Neat! :ok_hand:

Don't delay, go see the [Examples](https://github.com/sverrirs/jekyll-paginate-v2/tree/master/examples), they're way more useful than read-me docs at this point :)

## Backwards compatibility with jekyll-paginate
This gem is fully backwards compatible with the old [jekyll-paginate](https://github.com/jekyll/jekyll-paginate) gem and can be used as a zero-configuration replacement for it. If the old site config is detected then the gem will fall back to the old logic of pagination. 

> You cannot run both the new pagination logic and the old one at the same time

The following `_config.yml` settings are honored when running this gem in compatability mode

``` yml
paginate: 8
paginate_path: "/legacy/page:num/"
```

See more about the old style of pagination at the [jekyll-paginate](https://github.com/jekyll/jekyll-paginate) page.

> :bangbang: **Warning** Backwards compatibility with the old jekyll-paginate gem is currently scheduled to be removed after **1st January 2018**. Users will start receiving warning log messages when running jekyll two months before this date.

## Paginating collections
By default the pagination system only paginates `posts`. If you only have `posts` and `pages` in your site you don't need to worry about a thing, everything will work as intended without you configuring anything. 

However if you use document collections, or would like to, then this pagination gem offers extensive support for paginating documents in one or more collections at the same time. 

> Collections are groups of documents that belong together but should not be grouped by date. 
> See more about ['collections'](http://ben.balter.com/2015/02/20/jekyll-collections/) on Ben Balters blog.

### Paginating a single collection

Lets expand on Ben's collection discussion (linked above). Let's say that you have hundreds of cupcake pages in your cupcake collection. To create a pagination page for only documents from the cupcake collection you would do this

``` yml
---
layout: page
title: All Cupcakes
pagination: 
  enabled: true
  collection: cupcakes
---
```

### Paginating multiple collections

Lets say that you want to create a single pagination page for only small cakes on your page (you have both cupcakes and cookies to sell). You could do that like this

``` yml
---
layout: page
title: Lil'bits
pagination: 
  enabled: true
  collection: cupcakes, cookies
---
```

### The special 'all' collection

Now your site has grown and you have multiple cake collections on it and you want to have a single page that paginates all of your collections at the same time. 
You can use the special `all` collection name for this.

``` yml
---
layout: page
title: All the Cakes!
pagination: 
  enabled: true
  collection: all
---
```

> Note: Due to the `all` keyword being reserved for this feature, you cannot have a collection called `all` in your site configuration. Sorry. 


## Paginate categories, tags, locales

Enabling pagination for specific categories, tags or locales is as simple as adding values to the pagination page front-matter and corresponding values in the posts.

### Filtering categories

Filter single category 'software'

``` yml
---
layout: post
pagination: 
  enabled: true
  category: software
---
```

Filter multiple categories (lists only posts belonging to all categories)

``` yml
pagination: 
  enabled: true
  category: software, ruby
```

> To define categories you can either specify them in the front-matter or through the [directory structure](http://jekyllrb.com/docs/variables/#page-variables) of your jekyll site (Categories are derived from the directory structure above the \_posts directory). You can actually use both approaches to assign your pages to multiple categories.

### Filtering tags

Filter on a single tag

``` yml
pagination: 
  enabled: true
  tag: cool
```

Filter on multiple tags

``` yml
pagination: 
  enabled: true
  tag: cool, life
```

> When specifying tags in your posts make sure that the values are not enclosed in single quotes (double quotes are fine). If they are you will get a cryptic error when generating your site that looks like _"Error: could not read file <FILE>: did not find expected key while parsing a block mapping at line 2 column 1"_

### Filtering locales

In the case your site offers multiple languages you can include a `locale` item in your post front matter. The paginator can then use this value to filter on

The category page front-matter would look like this

``` yml
pagination: 
  enabled: true
  locale: en_US
```

Then for the relevant posts, include the `locale` variable in their front-matter

``` yml 
locale: en_US
```

## Paginate on combination of filters

Including only posts from categories 'ruby' and 'software' written in English

``` yml
pagination: 
  enabled: true
  category: software, ruby
  locale: en_US, en_GB, en_WW
```

Only showing posts tagged with 'cool' and in category 'cars'

``` yml
pagination: 
  enabled: true
  category: cars
  tag: cool
```

... and so on and so on

## Configuration overrides

All of the configuration elements from the `_config.yml` file can be overwritten in the pagination pages. E.g. if you want one category page to have different permalink structure simply override the item like so

``` yml
pagination: 
  enabled: true
  category: cars
  permalink: '/cars/:num/'
```

Overriding sorting to sort by the post title in ascending order for another paginated page could be done like so

``` yml
pagination: 
  enabled: true
  category: ruby
  sort_field: 'title'
  sort_reverse: false
```

## Advanced Sorting
Sorting can be done by any field that is available in the post front-matter. You can even sort by nested fields.

> When sorting by nested fields separate the fields with a colon `:` character.

As an example, assuming all your posts have the following front-matter

``` yml
---
layout: post
author:
  name: 
    first: "John"
    last: "Smith"
  born: 1960
---
```

You can define pagination sorting on the nested `first` field like so

``` yml
---
layout: page
title: "Authors by first name"
pagination: 
  enabled: true
  sort_field: 'author:name:first'
---
```

To sort by the `born` year in decending order (youngest first)

``` yml
---
layout: page
title: "Authors by birth year"
pagination: 
  enabled: true
  sort_field: 'author:born'
  sort_reverse: true
---
```

## Creating Pagination Trails

<p align="center">
  <img src="https://raw.githubusercontent.com/sverrirs/jekyll-paginate-v2/master/res/pagination-trails.png" />
</p>

Creating a trail structure for your pagination as shown above can be achieved by enabling the `trail` configuration and including a little extra code in your liquid templates.

``` yml
pagination:
  trail: 
    before: 2 # The number of links before the current page
    after: 2  # The number of links after the current page
```

Your layout file would then have to include code similar to the following to generate the correct HTML structure

``` HTML
{% if paginator.page_trail %}
  {% for trail in paginator.page_trail %}
    <li {% if page.url == trail.path %}class="selected"{% endif %}>
        <a href="{{ trail.path | prepend: site.baseurl }}" title="{{trail.title}}">{{ trail.num }}</a>
    </li>
  {% endfor %}
{% endif %}
```
_See [example 3](https://github.com/sverrirs/jekyll-paginate-v2/tree/master/examples/03-tags) for a demo of a pagination trail_

The `trail` object exposes three properties:
* `num`: The number of the page
* `path`: The path to the page
* `title`: The title of the page

The algorithm will always attempt to keep the same trail length for all pages (`trail length = before + after + 1`). 
As an example if we have only 7 pagination pages in total and the user is currently on page 6 then the trail would look like this

<p align="center">
  <img src="https://raw.githubusercontent.com/sverrirs/jekyll-paginate-v2/master/res/pagination-trails-p6.png" />
</p>

Different number of before and after trail links can be specified. Below is an example of how the yml config below would look like when on the same page 4

``` yml
pagination:
  trail: 
    before: 1
    after: 3
```

<p align="center">
  <img src="https://raw.githubusercontent.com/sverrirs/jekyll-paginate-v2/master/res/pagination-trails-p4-b1a3.png" />
</p>

## Detecting generated pagination pages

To identify the auto-generated pages that are created by the pagination logic when iterating through collections such as `site.pages` the `page.autogen` variable can be used like so

```
{% for my_page in site.pages %}
  {% if my_page.title and my_page.autogen == nil %}
    <h1>{{ my_page.title | escape }}</h1>
  {% endif %}
{% endfor %}
```
_In this example only pages that have a title and are not auto-generated are included._

This variable is created and assigned the value `page.autogen = "jekyll-paginate-v2"` by the pagination logic. This way you can detect which pages are auto-generated and by what gem. 

## Formatting page titles

The `title` field in both the site.config and the front-matter configuration supports the following macros.

| Text | Replaced with | Example |
| --- | --- | --- |
| :title | original page title | Page with `title: "Index"` and paginate config `title: ":title - split"` becomes `<title>Index - split</title>` |
| :num | number of the current page | Page with `title: "Index"` and paginate config `title: ":title (page :num)"` the second page becomes `<title>Index (page 2)</title>` |
| :max | total number of pages | Page with paginate config `title: ":num of :max"` the third page of 10 will become `<title>3 of 10</title>"` |

## Reading pagination meta information
Each pagination page defines an information structure `pagination_info` that is available to the liquid templates. This structure contains meta information for the pagination process, such as current pagination page and the total number of paginated pages.

The following fields are available

| Field | Description |
| --- | --- |
| curr_page | The number of the current pagination page |
| total_pages | The total number of pages in this pagination |

Below is an example on how to print out a "Page x of n" in the pagination layout

``` html
<h2>Page {{page.pagination_info.curr_page}} of {{page.pagination_info.total_pages}}</h2>
```

## Generating a JSON API

Delivering content via an API is useful, for a lot of the same reasons that pagination is useful. We want to delivery content, in such a way, that is:

1. Easy for the user to consume.
2. Easy for the browser to load.

Paginating content meets both of these requirements, but developers are limited to presenting content statically rather than dynamically. Some example of dynamic content delivery are:
- Pop up modals
- Infinite scrolling
- Multi-tiered pagination (e.g. Netflix UI horizontal scrolling for multiple movie categories)

### So how do I generate a JSON API for Jekyll?

First, create a new jekyll page and set its layout to `null` to avoid any extra html to show up.

Next, use the `extension` and `indexpage` option to customize the output of the page and its paginated content as JSON files.
> Note that the `indexpage` field also supports the same macros as the permalink field

Here's an example page:
```
---
layout: null
permalink: /api
pagination:
  permalink: ''
  enabled: true
  extension: .json
  indexpage: 'feed-:num'
---

{
  "pages": [{% for post in paginator.posts %}
    {% if forloop.first != true %},{% endif %}
    {
      "title": "{{ post.title }}",
      "link": "{{ post.url }}"
    }{% endfor %}
  ]
}
```
Next, run `jekyll build`. This will generate a set of paginated JSON files under the folder `/api`. These JSON files can be loaded via Javascript/AJAX to dynamically load content into your site.

Below's an example set of routes that the configuration would generate:
- http://localhost:4000/api/feed-1.json
- http://localhost:4000/api/feed-2.json
- http://localhost:4000/api/feed-3.json

And here is an example of one of the feed.json files that are created given the markup above
```
{
  "pages": [
    {
      "title": "Narcisse Snake Pits",
      "link": "/2016/11/narcisse-snake-pits.html"
    },{
      "title": "Luft-Fahrzeug-Gesellschaft",
      "link": "/2016/11/luft-fahrzeug-gesellschaft.html"
    },{
      "title": "Rotary engine",
      "link": "/2016/11/rotary-engine.html"
    }
  ], 
  "next": "/api/feed-11.json",
  "prev": "/api/feed-9.json",
  "first": "/api/feed-1.json"
}
```

For further information see [Example 4](https://github.com/sverrirs/jekyll-paginate-v2/tree/master/examples/04-jsonapi), that project can serve as a starting point for your experiments with this feature.

### How did you generate those 'next', 'prev' and 'first' links?

All the normal paginator variables can be used in these JSON feed files. You can use them to achive quite powerful features such as pre-loading and detecting when there are no more feeds to load. 

```
{% if paginator.next_page %}
  ,"next": "{{ paginator.next_page_path }}"
  {% endif %}
  {% if paginator.last_page %}
  ,"prev": "{{ paginator.last_page_path }}"
  {% endif %}
  {% if paginator.first_page %}
  ,"first": "{{ paginator.first_page_path }}"
  {% endif %}
```

## Renaming pagination file names
By default the pagination system creates all paginated pages as `index.html`. The system provides an option to override this name and file extension with the 

```yml
  indexpage: index
  extension: html
```

If you wanted to generate all pagination files as `default.htm` then the settings should be configured as follows

```yml
  indexpage: default
  extension: htm
```

## Common issues

### I keep getting a dependency error when running jekyll serve after installing this gem

> Dependency Error: Yikes! It looks like you don't have jekyll-paginate-v2 or one of its dependencies installed...

Check your `Gemfile` in the site root. Ensure that the jekyll-paginate-v2 gem is present in the jekyll_plugins group like the example below. If this group is missing add to the file.

``` ruby
group :jekyll_plugins do
  gem "jekyll-paginate-v2"
end
```

### I'm getting a bundler error after upgrading the gem (Bundler::GemNotFound)

> bundler/spec_set.rb:95:in `block in materialize': Could not find jekyll-paginate-v2-1.0.0 in any of the sources (Bundler::GemNotFound)

Delete your `Gemfile.lock` file and try again.


### I'm getting a bundler error after upgrading the gem (Gem::LoadError)

> bundler/runtime.rb:40:in 'block in setup': You have already activated addressable 2.5.0, but your Gemfile requires addressable 2.4.0. Prepending `bundle exec` to your command may solve this. (Gem::LoadError)

Delete your `Gemfile.lock` file and try again.

### My pagination pages are not being found (Couldn't find any pagination page. Skipping pagination)

> Pagination: Is enabled, but I couldn't find any pagination page. Skipping pagination...

* Ensure that you have the correct minimum front-matter in the pagination pages
``` yml
pagination:
  enabled: true
```
* You can place pagination logic into either the pages or liquid templates (templates are stored under the `_layouts/` and `_includes/` folders).

### My pages are being nested multiple levels deep

When using `categories` for posts it is advisable to explicitly state a `permalink` structure in your `_config.yml` file. 

```
permalink: /:year/:month/:title.html
```

This is because the default behavior in Jekyll is to nest pages for every category that they belong to and Jekyll unfortunately does not understand multi-categories separated with `,` or `;` but instead does all separation on `[space]` only. 

### My pagination pages are overwriting each others pages
If you specify multiple pages that paginate in the site root then you must give them unique and separate pagination permalink. This link is set in the pagination page front-matter like so

``` yml
pagination:
  enabled: true
  permalink: '/cars/:num/'
```

Make absolutely sure that your pagination permalink paths do not clash with any other paths in your final site. For simplicity it is recommended that you keep all custom pagination (non root index.html) in a single or multiple separate sub folders under your site root.
