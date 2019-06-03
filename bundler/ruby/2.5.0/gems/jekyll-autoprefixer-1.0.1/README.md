# Jekyll Autoprefixer [![Gem Version](https://badge.fury.io/rb/jekyll-autoprefixer.png)](http://badge.fury.io/rb/jekyll-autoprefixer)

> Autoprefixer integration for Jekyll

This plugin provides simple autoprefixer support for Jekyll.

## Installation

This plugin is available as a [RubyGem][ruby-gem].

Add this line to your application's `Gemfile`:

```
gem 'jekyll-autoprefixer'
```

And then execute the `bundle` command to install the gem.

Alternatively, you can also manually install the gem using the following command:

```
$ gem install jekyll-autoprefixer
```

After the plugin has been installed successfully, add the following lines to your `_config.yml` in order to tell Jekyll to use the plugin:

```
gems:
- jekyll-autoprefixer
```

## Getting Started

No additional steps are required. All written CSS files inside the destination
directory are overwritten with the output of autoprefixer.

Optionally, you can specify the browsers for which autoprefixer is supposed to generate prefixes inside your configuration:

```
autoprefixer:
  browsers:
  - last 2 versions
```

You can also specify that autoprefixer should only work in production mode:

```
autoprefixer:
  only_production: true
```

# Contribute

Fork this repository, make your changes and then issue a pull request. If you find bugs or have new ideas that you do not want to implement yourself, file a bug report.

# Copyright

Copyright (c) 2015 Vincent Wochnik.

License: MIT

[ruby-gem]: https://rubygems.org/gems/jekyll-email-protect
