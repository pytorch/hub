# Jekyll::Paginate V2

Pagination gem built specially for Jekyll 3 and newer that is fully backwards compatable and serves as an enhanced replacement for the previously built-in [jekyll-paginate gem](https://github.com/jekyll/jekyll-paginate). View it on [rubygems.org](https://rubygems.org/gems/jekyll-paginate-v2).

[![Gem](https://img.shields.io/gem/v/jekyll-paginate-v2.svg)](https://rubygems.org/gems/jekyll-paginate-v2)
[![Join the chat at https://gitter.im/jekyll-paginate-v2/Lobby](https://badges.gitter.im/jekyll-paginate-v2/Lobby.svg)](https://gitter.im/jekyll-paginate-v2/Lobby)
[![Build Status](https://travis-ci.org/sverrirs/jekyll-paginate-v2.svg?branch=master)](https://travis-ci.org/sverrirs/jekyll-paginate-v2) 
[![Dependency Status](https://gemnasium.com/badges/github.com/sverrirs/jekyll-paginate-v2.svg)](https://gemnasium.com/github.com/sverrirs/jekyll-paginate-v2)
[![Code Climate](https://codeclimate.com/github/sverrirs/jekyll-paginate-v2/badges/gpa.svg)](https://codeclimate.com/github/sverrirs/jekyll-paginate-v2)
[![security](https://hakiri.io/github/sverrirs/jekyll-paginate-v2/master.svg)](https://hakiri.io/github/sverrirs/jekyll-paginate-v2/master)
[![Gem](https://img.shields.io/gem/dt/jekyll-paginate-v2.svg)](https://rubygems.org/gems/jekyll-paginate-v2)

Reach me at the [project issues](https://github.com/sverrirs/jekyll-paginate-v2/issues) section or via email at [jekyll@sverrirs.com](mailto:jekyll@sverrirs.com), you can also get in touch on the project's [Glitter chat room](https://gitter.im/jekyll-paginate-v2/Lobby).

> The code was based on the original design of [jekyll-paginate](https://github.com/jekyll/jekyll-paginate) and features were sourced from discussions such as [#27](https://github.com/jekyll/jekyll-paginate/issues/27) (thanks [Günter Kits](https://github.com/gynter)).

* [Installation](#installation)
* [Example Sites](https://github.com/sverrirs/jekyll-paginate-v2/tree/master/examples)
* [Pagination Generator](#pagination-generator)
* [Auto-Pages](#auto-pages)
* [Issues / to-be-completed](#issues--to-be-completed)
* [How to Contribute](#contributing)

> _"Be excellent to each other"_

:heart:

## Installation

```
gem install jekyll-paginate-v2
```

Update your [_config.yml](README-GENERATOR.md#site-configuration) and [pages](README-GENERATOR.md#page-configuration).

> Although fully backwards compatible, to enable the new features this gem needs slightly extended [site yml](README-GENERATOR.md#site-configuration) configuration and miniscule additional new front-matter for the [pages to paginate on](README-GENERATOR.md#page-configuration).

Now you're ready to run `jekyll serve` and your paginated files should be generated.

Please see the [Examples](https://github.com/sverrirs/jekyll-paginate-v2/tree/master/examples) for tips and tricks on how to configure the pagination logic.


## Pagination Generator

The [Pagination Generator](README-GENERATOR.md) forms the core of the pagination logic. Calculates and generates the pagination pages.


## Auto Pages

The [Auto-Pages](README-AUTOPAGES.md) is an optional feature that auto-magically generates paginated pages for all your tags, categories and collections.


## Issues / to-be-completed

* Unit-tests do not cover all critical code paths
* No integration tests yet [#2](https://github.com/jekyll/jekyll-paginate/pull/2)
* _Exclude_ filter not implemented [#6](https://github.com/jekyll/jekyll-paginate/issues/6)
* Elegant way of collecting and printing debug information during pagination


I welcome all testers and people willing to give me feedback and code reviews.

## Contributing

> Although this project is small it has a [code of conduct](CODE_OF_CONDUCT.md) that I hope everyone will do their best to follow when contributing to any aspects of this project. Be it discussions, issue reporting, documentation or programming. 

If you don't want to open issues here on Github, send me your feedback by email at [jekyll@sverrirs.com](mailto:jekyll@sverrirs.com).

1. Fork it ( https://github.com/sverrirs/jekyll-paginate-v2/fork )
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Run the unit tests (`rake`)
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Build the gem locally (`gem build jekyll-paginate-v2.gemspec`)
6. Test and verify the gem locally (`gem install ./jekyll-paginate-v2-x.x.x.gem`) 
7. Push to the branch (`git push origin my-new-feature`)
8. Create new Pull Request

Note: This project uses [semantic versioning](http://semver.org/).
