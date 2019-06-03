require_relative '../spec_helper.rb'

module Jekyll::PaginateV2::Generator
  describe Utils do

    it "should always replace num format with the specified number" do
      Utils.format_page_number( ":num", 7).must_equal "7"
      Utils.format_page_number( ":num", 13).must_equal "13"
      Utils.format_page_number( ":num", -2).must_equal "-2"
      Utils.format_page_number( ":num", 0).must_equal "0"
      Utils.format_page_number( ":num", 1000).must_equal "1000"
    end

    it "should always replace num format with the specified number and keep rest of formatting" do
      Utils.format_page_number( "/page:num/", 7).must_equal "/page7/"
      Utils.format_page_number( "/page:num/", 50).must_equal "/page50/"
      Utils.format_page_number( "/page:num/", -5).must_equal "/page-5/"

      Utils.format_page_number( "/car/:num/", 1).must_equal "/car/1/"
      Utils.format_page_number( "/car/:num", 1).must_equal "/car/1"
      Utils.format_page_number( "car/:num", 1).must_equal "car/1"
      Utils.format_page_number( "/car//:num", 1).must_equal "/car//1"
    end

    it "make sure there is a leading slash in path" do
      Utils.ensure_leading_slash("path/to/file/wow").must_equal "/path/to/file/wow"
      Utils.ensure_leading_slash("/no/place/wow/").must_equal "/no/place/wow/"
      Utils.ensure_leading_slash("/no").must_equal "/no"
      Utils.ensure_leading_slash("no").must_equal "/no"
    end    

    it "make sure there is never a leading slash in path" do
      Utils.remove_leading_slash("path/to/file/wow").must_equal "path/to/file/wow"
      Utils.remove_leading_slash("/no/place/wow/").must_equal "no/place/wow/"
      Utils.remove_leading_slash("/no").must_equal "no"
      Utils.remove_leading_slash("no").must_equal "no"
    end

    it "sort must sort strings lowercase" do
      Utils.sort_values( "AARON", "Aaron").must_equal 0
      Utils.sort_values( "AARON", "aaron").must_equal 0
      Utils.sort_values( "aaron", "AARON").must_equal 0 
    end

    it "when sorting by nested post data the values must be resolved fully" do
      data = {'book'=>{ 'name' => { 'first'=> 'John', 'last'=> 'Smith'}, 'rank'=>20}}
      Utils.sort_get_post_data(data, "book:rank").must_equal 20
      Utils.sort_get_post_data(data, "book:name:first").must_equal "John"
      Utils.sort_get_post_data(data, "book:name:last").must_equal "Smith"

      Utils.sort_get_post_data(data, "book:name").must_be_nil
      Utils.sort_get_post_data(data, "name").must_be_nil
      Utils.sort_get_post_data(data, "book").must_be_nil
    end

    it "should always replace max format with the specified number if specified" do
      Utils.format_page_number( ":num-:max", 7, 16).must_equal "7-16"
      Utils.format_page_number( ":num-:max", 13, 20).must_equal "13-20"
      Utils.format_page_number( ":num-:max", -2, -4).must_equal "-2--4"
      Utils.format_page_number( ":num_of_:max", 0, 10).must_equal "0_of_10"
      Utils.format_page_number( ":num/:max", 1000, 2000).must_equal "1000/2000"
    end


  end
end

