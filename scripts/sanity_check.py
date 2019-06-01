import argparse
import copy
import os
import glob
from urllib.request import urlopen, HTTPError


class ValidMD:
    def __init__(self, filename):
        self.filename = filename
        self.required_user_fields = ['title', 'summary', 'image', 'author', 'tags',
                        'github-link', 'category']

        self.optional_image_fields = ['featured_image_1', 'featured_image_2']

        self.valid_tags = ['vision', 'text', 'audio']

        self.valid_categories = ['researchers', 'developers']

        self.required_headers_untouched = """
---
layout: pytorch_hub_detail
background-class: hub-background
body-class: hub"""

        self.required_sections = ['Model Description', 'Example']


    def validate_tags(self, tags):
        '''
        Only allow tags in predefined set
        '''
        if tags.startswith('['):
            tags = tags[1:-1].split(',')
        elif ',' in tags:
            raise ValueError('Mulple tags {} must be surrounded by [] in file {}'
                             .format(tags, self.filename))
        else:
            tags = [tags]
        for t in tags:
            if t not in self.valid_tags:
                continue
                # FIXME: Enable this when tags are cleaned up
                # raise ValueError('Tag {} is not valid in {}. Valid tag set is {}'
                                 # .format(t, self.filename, self.valid_tags))


    def validate_category(self, category):
        '''
        Only allow categories in predefined set
        '''
        if category not in self.valid_categories:
            raise ValueError('Category {} is not valid in {}. Choose from {}'
                             .format(category, self.filename, self.valid_categories))


    def validate_github_link(self, link):
        '''
        Make sure the github repo exists
        '''
        try:
            ret = urlopen(link)
        except HTTPError:
            raise ValueError('{} is not valid url in {}'.format(link, self.filename))


    def validate_image(self, image_name):
        '''
        Make sure reference image exists in images/
        '''
        images = [os.path.basename(i) for i in glob.glob('images/*')]\
            + ['pytorch-logo.png', 'no-image']
        if image_name not in images:
            raise ValueError('Image {} referenced in {} not found in images/'.format(image_name, self.filename))


    def validate_required_headers(self, headers):
        '''
        Make sure required headers are untouched
        '''
        for h in headers:
            if h.strip() not in self.required_headers_untouched:
                raise ValueError(
                        'File {} must start with these lines untouched:\n {}'
                        .format(self.filename, self.required_headers_untouched))


    def no_extra_colon(self, value):
        if ':' in value:
             raise ValueError('Remove \':\' in field {} in file {}'.format(value, self.filename))


    def check_markdown_file(self):
        print('Checking {}...'.format(self.filename))
        with open(self.filename, 'r') as f:
            len_required = len(self.required_headers_untouched.split('\n')) - 1
            headers = [next(f) for x in range(len_required)]
            self.validate_required_headers(headers)
            for line in f:
                if ':' in line:
                    field, value = [x.strip() for x in line.split(':', 1)]
                    if field in self.required_user_fields:
                        if field == 'tags':
                            self.validate_tags(value)
                        elif field == 'github-link':
                            self.validate_github_link(value)
                        elif field == 'image':
                            self.validate_image(value)
                        elif field == 'category':
                            self.validate_category(value)
                        else:
                            # Jekyll doesn't build with extra colon in these fields
                            self.no_extra_colon(value)
                        self.required_user_fields.remove(field)
                    elif field in self.optional_image_fields:
                        self.validate_image(value)

                if line.startswith('###'):
                    # No we don't want colon after section names
                    self.no_extra_colon(line)
                    _, section = [x.strip() for x in line.split(' ', 1)]
                    if section in self.required_sections:
                        self.required_sections.remove(section)

            if len(self.required_user_fields) != 0:
                raise ValueError('Missing required field {} in file {}'
                                 .format(self.required_user_fields, self.filename))

            if len(self.required_sections) != 0:
                raise ValueError('Missing required section {} in file {}'
                                 .format(self.required_sections, self.filename))


def sanity_check():
    for f in glob.glob('*.md'):
        # Skip README
        if f == 'README.md':
            continue
        ValidMD(f).check_markdown_file()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default=None, help='filename')
    args = parser.parse_args()
    if args.file:
        ValidMD(args.file).check_markdown_file()
    else:
        sanity_check()
