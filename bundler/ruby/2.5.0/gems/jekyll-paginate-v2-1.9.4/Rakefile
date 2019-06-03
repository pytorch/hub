require 'rake'
require 'rake/testtask'

#$LOAD_PATH.unshift(File.join(File.dirname(__FILE__), *%w[lib]))
#require 'lib/jekyll-paginate-v2/version'

Rake::TestTask.new do |t|
  t.libs.push 'lib'
  t.libs.push 'specs'
  t.verbose = true
  t.pattern = "spec/**/*_spec.rb"
  t.test_files = FileList['spec/**/*_spec.rb']
end

desc "Run tests"
task :default => [:test]