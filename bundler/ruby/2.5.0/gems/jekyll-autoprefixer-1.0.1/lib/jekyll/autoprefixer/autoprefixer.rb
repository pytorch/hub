# Frozen-string-literal: true
# Encoding: utf-8

require 'autoprefixer-rails'

module Jekyll
  module Autoprefixer
    class Autoprefixer
      attr_reader :site, :batch

      def initialize(site)
        @site = site
        @batch = Array.new
      end

      def process()
        options = @site.config['autoprefixer'] || {}

        if !options['only_production'] || Jekyll.env == "production"
          @batch.each do |item|
            path = item.destination(@site.dest)

            File.open(path, 'r+') do |file|
              content = file.read
              file.truncate(0)
              file.rewind
              file.write(AutoprefixerRails.process(content, options))
            end
          end
        end

        @batch.clear
      end
    end
  end
end
