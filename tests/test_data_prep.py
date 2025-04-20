import unittest

class test_data_prep(unittest.TestCase):

    def test_chunk_terms_service(self):
        from data_prep.core import chunk_terms_service
        file = r"F:\data\terms-service\text\1LikeNoOther_PrivacyPolicy.txt"
        with open(file, 'r') as f:
            contents = f.read()
        cleaned = chunk_terms_service(contents)
        print(len(cleaned))

    def test_parse_text_dataset(self):
        pass

