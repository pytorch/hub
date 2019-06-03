# Jekyll::Paginate V2 is a gem built for Jekyll 3 that generates pagiatation for posts, collections, categories and tags.
# 
# It is based on https://github.com/jekyll/jekyll-paginate, the original Jekyll paginator
# which was decommissioned in Jekyll 3 release onwards. This code is currently not officially
# supported on Jekyll versions < 3.0 (although it might work)
#
# Author: Sverrir Sigmundarson
# Site: https://github.com/sverrirs/jekyll-paginate-v2
# Distributed Under The MIT License (MIT) as described in the LICENSE file
#   - https://opensource.org/licenses/MIT

require "jekyll-paginate-v2/version"
# Files needed for the pagination generator
require "jekyll-paginate-v2/generator/defaults"
require "jekyll-paginate-v2/generator/compatibilityUtils"
require "jekyll-paginate-v2/generator/utils"
require "jekyll-paginate-v2/generator/paginationIndexer"
require "jekyll-paginate-v2/generator/paginator"
require "jekyll-paginate-v2/generator/paginationPage"
require "jekyll-paginate-v2/generator/paginationModel"
require "jekyll-paginate-v2/generator/paginationGenerator"
# Files needed for the auto category and tag pages
require "jekyll-paginate-v2/autopages/utils"
require "jekyll-paginate-v2/autopages/defaults"
require "jekyll-paginate-v2/autopages/autoPages"
require "jekyll-paginate-v2/autopages/pages/baseAutoPage"
require "jekyll-paginate-v2/autopages/pages/categoryAutoPage"
require "jekyll-paginate-v2/autopages/pages/collectionAutoPage"
require "jekyll-paginate-v2/autopages/pages/tagAutoPage"

module Jekyll 
  module PaginateV2
  end # module PaginateV2
end # module Jekyll