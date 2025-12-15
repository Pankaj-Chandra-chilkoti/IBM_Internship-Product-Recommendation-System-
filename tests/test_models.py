# tests/test_models.py

import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask
from models import db, User

class TestDatabaseModels(unittest.TestCase):
    def setUp(self):
        
        self.app = Flask(__name__)
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        db.init_app(self.app)

        with self.app.app_context():
            db.create_all()  

    def test_create_user(self):
        with self.app.app_context():
            user = User(username='pankaj', email='pankaj@example.com')
            db.session.add(user)
            db.session.commit()

            queried_user = User.query.filter_by(username='pankaj').first()
            self.assertIsNotNone(queried_user)
            self.assertEqual(queried_user.email, 'pankaj@example.com')

if __name__ == '__main__':
    unittest.main()
