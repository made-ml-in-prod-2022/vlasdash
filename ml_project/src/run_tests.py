import unittest


class TestAll(unittest.TestCase):

    def run_all(self) -> None:
        loader = unittest.TestLoader()
        start_dir = 'tests'
        tests = loader.discover(start_dir)
        runner = unittest.TextTestRunner()
        runner.run(tests)


if __name__ == "__main__":
    unittest.main()
