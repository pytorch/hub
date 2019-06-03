# Frozen-string-literal: true
# Encoding: utf-8

require 'jekyll'

require_relative 'jekyll/patches'
require_relative 'jekyll/autoprefixer'

Jekyll::Hooks.register :site, :after_reset do |site|
  # create new autoprefixer instance
  site.autoprefixer = Jekyll::Autoprefixer::Autoprefixer.new(site)
end

Jekyll::Hooks.register :site, :post_render do |site|
  site.each_site_file do |item|
    if site.regenerator.regenerate?(item)
      ext = File.extname(item.destination(site.dest))
      site.autoprefixer.batch.push(item) if ext == ".css"
    end
  end
end

Jekyll::Hooks.register :site, :post_write do |site|
  site.autoprefixer.process
end
