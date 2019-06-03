require_relative '../spec_helper.rb'

module Jekyll::PaginateV2::Generator
  describe "checking default config" do

    it "should always contain the following keys" do
      DEFAULT.must_include 'enabled'
      DEFAULT.must_include 'collection'
      DEFAULT.must_include 'per_page'
      DEFAULT.must_include 'permalink'
      DEFAULT.must_include 'title'
      DEFAULT.must_include 'page_num'
      DEFAULT.must_include 'sort_reverse'
      DEFAULT.must_include 'sort_field'
      DEFAULT.must_include 'limit'
      DEFAULT.must_include 'debug'
      DEFAULT.size.must_be :>=, 10
    end

    it "should always contain the following key defaults" do
      DEFAULT['enabled'].must_equal false
      DEFAULT['collection'].must_equal 'posts'
      DEFAULT['per_page'].must_equal 10
      DEFAULT['permalink'].must_equal '/page:num/'
      DEFAULT['title'].must_equal ':title - page :num'
      DEFAULT['page_num'].must_equal 1
      DEFAULT['sort_reverse'].must_equal false
      DEFAULT['sort_field'].must_equal 'date'
      DEFAULT['limit'].must_equal 0
      DEFAULT['debug'].must_equal false
    end

  end
end