import argparse
import os
import glob
from urllib.request import urlopen, HTTPError
from tags import valid_tags
import yaml
import mistune

class ValidMD:
    def __init__(self, filename):
        self.filename = filename
        self.required_user_fields = ['title', 'summary', 'image', 'author',
                                     'tags', 'github-link', 'category']

        self.optional_image_fields = ['featured_image_1', 'featured_image_2']

        self.valid_tags = valid_tags

        self.valid_categories = ['researchers', 'developers']

        self.required_sections = ['Model Description']

        self.optional_demo_link = ['demo-model-link']

    def validate_tags(self, tags):
        '''
        Only allow tags in pre-defined set
        '''
        for t in tags:
            if t not in self.valid_tags:
                raise ValueError(
                    'Tag {} is not valid in {}. Valid tag set is {}'
                    .format(t, self.filename, self.valid_tags))

    def validate_category(self, category):
        '''
        Only allow categories in predefined set
        '''
        if category not in self.valid_categories:
            raise ValueError(
                    'Category {} is not valid in {}. Choose from {}'
                    .format(category, self.filename, self.valid_categories))

    def validate_link(self, link):
        '''
        Make sure the github repo exists
        '''
        try:
            urlopen(link)
        except HTTPError:
            raise ValueError('{} is not valid url in {}'
                             .format(link, self.filename))

    def validate_image(self, image_name):
        '''
        Make sure reference image exists in images/
        '''
        images = [os.path.basename(i) for i in glob.glob('images/*')]\
            + ['pytorch-logo.png', 'no-image']
        if image_name not in images:
            raise ValueError('Image {} referenced in {} not found in images/'
                             .format(image_name, self.filename))

    def validate_header(self, header):
        '''
        Make sure the header is in the required format
        '''
        assert header['layout'] == 'hub_detail'
        assert header['background-class'] == 'hub-background'
        assert header['body-class'] == 'hub'

        for field in self.required_user_fields:
            header[field]  # assert that it exists

        self.validate_tags(header['tags'])
        self.validate_link(header['github-link'])
        self.validate_image(header['image'])
        self.validate_category(header['category'])

        for field in self.optional_demo_link:
            if field in header.keys():
                self.validate_link(header[field])

        for field in self.optional_image_fields:
            if field in header.keys():
                self.validate_image(header[field])

        for k in header.keys():
            if not k.endswith('-link'):
                self.no_extra_colon(k, header[k])


    def no_extra_colon(self, field, value):
        # Jekyll doesn't build with extra colon in these fields
        if ':' in str(value):
            raise ValueError('Remove extra \':\' in field {} with value {} in file {}'
                             .format(field, value, self.filename))

    def validate_markdown(self, markdown):
        m = mistune.create_markdown(renderer=mistune.AstRenderer())

        for block in m(markdown):
            if block['type'] == 'heading':
                # we dont want colon after section names
                text_children = [c for c in block['children'] if c['type'] == 'text']
                for c in text_children:
                    assert not c['text'].endswith(':')
                    if c['text'] in self.required_sections:
                        self.required_sections.remove(c['text'])
        try:
            assert len(self.required_sections) == 0
        except AssertionError as e:
            print("Missing required sections: {}".format(self.required_sections))
            raise e


    def check_markdown_file(self):
        print('Checking {}...'.format(self.filename))

        # separate header and markdown.
        # Then, check header and markdown separately
        header = []
        markdown = []
        header_read = False
        with open(self.filename, 'r') as f:
            for line in f:
                if line.startswith('---'):
                    header_read = not header_read
                    continue
                if header_read == True:
                    header += [line]
                else:
                    markdown += [line]

        # checks that it's valid yamp
        header = yaml.safe_load(''.join(header))
        assert header, "Failed to parse a valid yaml header"
        self.validate_header(header)


        # check markdown
        markdown = "".join(markdown)
        self.validate_markdown(markdown)

def sanity_check():
    for f in glob.glob('*.md'):
        # Skip documentation
        if f in ('README.md', 'CONTRIBUTING.md', 'CODE_OF_CONDUCT.md'):
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
