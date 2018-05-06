import unittest
import os
from bs4 import BeautifulSoup

from util import is_validated, load_labeled_data
import dirs


class TestUtil(unittest.TestCase):

    def test_is_subsumed(self):
        dom = "<input autocomplete='given-name' id='firstname' maxlength='15' name='firstname' type='text' value=''/>"
        soup = BeautifulSoup(dom, 'html5lib')
        soup = soup.find('input', attrs={'type': 'text'})
        for pattern in ['firstname', '\w+name', 'f.*name', 'first[-_]?name']:
            self.assertTrue(is_validated(soup, pattern))

        dom = "<input autocomplete='given-name' id='first name' maxlength='15' name='firstname' type='text' value=''/>"
        soup = BeautifulSoup(dom, 'html5lib')
        soup = soup.find('input', attrs={'type': 'text'})
        self.assertTrue(is_validated(soup, 'first'))
        self.assertTrue(is_validated(soup, 'first name'))
        self.assertTrue(is_validated(soup, 'first.*name'))

        dom = "<input autocomplete='given-name' id='first-name' maxlength='15' name='firstname' type='text' value=''/>"
        soup = BeautifulSoup(dom, 'html5lib')
        soup = soup.find('input', attrs={'type': 'text'})
        self.assertTrue(is_validated(soup, 'first'))
        self.assertTrue(is_validated(soup, 'first-name'))
        self.assertTrue(is_validated(soup, 'first[-_]?name'))

        dom = """
        <input afv-validate="true" class="ng-isolate-scope ng-pristine ng-valid-pattern ng-invalid ng-invalid-required" data-server="" data-val="true" data-val-length="'first name' must be between 0 and 30 characters. you entered {totallength} characters." data-val-length-max="30" data-val-regex="'first name' is not in the correct format." data-val-regex-pattern="^[^&lt;&gt;;%!#()$^]*$" data-val-required="'first name' should not be empty." id="datamodel-shippingcontact-firstname" maxlength="30" name="datamodel-shippingcontact-firstname" ng-model="datamodel.shippingcontact.firstname" ng-pattern="/^[^&lt;&gt;;%!#()$^]*$/" required="" type="text" value=""/>
        """
        soup = BeautifulSoup(dom, 'html5lib')
        soup = soup.find('input', attrs={'type': 'text'})
        self.assertTrue(is_validated(soup, 'first'))

    def test_load_labeled_data(self):
        if __name__ == '__main__':
            data = load_labeled_data(dirs.parsed_dir)
            print(len(data))
            print(data[:3])


if __name__ == '__main__':
    unittest.main()
