from tunix.rl import utils
from absl.testing import absltest
import numpy as np

class UtilsTest(absltest.TestCase):
    
    def test_is_positive_integer(self):
        # 1. Standard Ints (Should Pass)
        try:
            utils.is_positive_integer(5, "test_var")
            utils.is_positive_integer(1, "test_var")
        except ValueError:
            self.fail("is_positive_integer raised ValueError unexpectedly for int!")

        # 2. Numpy Types (THIS IS THE BUG - It will fail now)
        try:
            utils.is_positive_integer(np.int64(5), "numpy_int")
            utils.is_positive_integer(np.float32(5.0), "numpy_float")
        except AttributeError:
             # We catch AttributeError specifically because that's what we are fixing
             self.fail("CRASH: AttributeError detected! The bug exists.")
        except ValueError:
            self.fail("is_positive_integer failed on Numpy types!")
        
        # 3. Fail Cases (Should raise ValueError)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            utils.is_positive_integer(5.5, "float_var")
            
        with self.assertRaisesRegex(ValueError, "positive integer"):
            utils.is_positive_integer(-5, "neg_var")

if __name__ == '__main__':
    absltest.main()