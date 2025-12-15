# tests/test_routes.py

import unittest
from app import app

class TestFlaskRoutes(unittest.TestCase):
    def setUp(self):
        
        self.app = app.test_client()
        self.app.testing = True

    def test_login_page(self):
        response = self.app.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data) 

    def test_register_page(self):
        response = self.app.get('/register')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Register', response.data)

    def test_homepage(self):
        
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)  
        self.assertIn(b'Trending', response.data)    

    def test_main_route(self):
        
        response = self.app.get('/main')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Recommendations', response.data)  

    def test_info_route(self):
        response = self.app.get('/info')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'About', response.data) 

if __name__ == '__main__':
    unittest.main()
