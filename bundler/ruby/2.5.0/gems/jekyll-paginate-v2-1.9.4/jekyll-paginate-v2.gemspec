# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'jekyll-paginate-v2/version'
require 'date'

Gem::Specification.new do |spec|
  spec.name          = "jekyll-paginate-v2"
  spec.version       = Jekyll::PaginateV2::VERSION
  spec.platform      = Gem::Platform::RUBY
  spec.required_ruby_version = '>= 2.0.0'  # Same as Jekyll
  spec.date          = DateTime.now.strftime('%Y-%m-%d')
  spec.authors       = ["Sverrir Sigmundarson"]
  spec.email         = ["jekyll@sverrirs.com"]
  spec.homepage      = "https://github.com/sverrirs/jekyll-paginate-v2"
  spec.license       = "MIT"

  spec.summary       = %q{Pagination Generator for Jekyll 3}
  spec.description   = %q{An enhanced zero-configuration in-place replacement for the now decomissioned built-in jekyll-paginate gem. This pagination gem offers full backwards compatability as well as a slew of new frequently requested features with minimal additional site and page configuration. Optional features include auto-generation of paginated collection, tag and category pages.}
  
  spec.files          = Dir['CODE_OF_CONDUCT.md', 'README*.md', 'LICENSE', 'Rakefile', '*.gemspec', 'Gemfile', 'lib/**/*', 'spec/**/*']
  spec.test_files    = spec.files.grep(%r{^(test|spec|features)/})
  spec.require_paths = ["lib"]

  # Gem requires Jekyll to work
  # ~> is the pessimistic operator and is equivalent to '>= 3.0', '< 4.0'
  spec.add_runtime_dependency "jekyll", "~> 3.0"

  # Development requires more
  spec.add_development_dependency "bundler", "~> 1.5"
  spec.add_development_dependency "rake", "~> 10.4"
  spec.add_development_dependency "minitest", '~> 5.4'
end