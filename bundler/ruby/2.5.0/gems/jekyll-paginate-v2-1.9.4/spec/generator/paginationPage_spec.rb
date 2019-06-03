require_relative '../spec_helper.rb'

module Jekyll::PaginateV2::Generator
  describe "tesing pagination page implementation" do

    it "sould always read the template file into itself" do
      # DUE TO THE JEKYLL:PAGE CLASS ACCESSING FILE IO DIRECTLY 
      # I AM UNABLE TO MOCK OUT THE FILE OPERATIONS TO CREATE UNIT TESTS FOR THIS CLASS :/
    end

  end
end