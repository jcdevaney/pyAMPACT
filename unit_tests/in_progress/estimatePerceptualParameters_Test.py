import unittest
import numpy as np

import sys
import os
sys.path.append(os.pardir)
from pitch import estimate_perceptual_parameters

class TestEstimatePerceptualParameters(unittest.TestCase):

    def test_estimate_perceptual_parameters(self):
        # Test data
        f0_vals = np.array([220.0, 330.0, 440.0, 330.0, 220.0])
        pwr_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        F = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        M = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.10, 0.11, 0.12], [0.13, 0.14, 0.15]])
        SR = 44100
        hop = 32
        gt_flag = False

        # Call the function
        result = estimate_perceptual_parameters(f0_vals, pwr_vals, F, M, SR, hop, gt_flag, 1)


        # print(result["ppitch"][0])
        # print(result["ppitch"][1])
        # print(result["jitter"])
        # print(result["vibrato_depth"])
        # print(result["vibrato_rate"])
        # print(result["shimmer"])
        # print(result["pwr_vals"])                    
        # print(result["f0_vals"])                                            
        # print(result["spec_centroid"])                 
        # print(result["spec_slope"])                                                                            
        # print(result["mean_spec_slope"])        
        # print(result["spec_flux"])                            
        # print(result["mean_spec_flux"])
        # print(result["spec_flat"])                    
        # print(result["mean_spec_flat"])

                               
        # TESTS
        self.assertAlmostEqual(result["ppitch"][0], 220.0, delta=0.01)
        self.assertAlmostEqual(result["jitter"], 110.0, delta=0.01)
        self.assertAlmostEqual(result["vibrato_depth"], 115.19, delta=0.01)
        self.assertAlmostEqual(result["vibrato_rate"], 275.625, delta=0.01)
        self.assertAlmostEqual(result["shimmer"], 5.19, delta=0.01)

        for pair in zip(result["pwr_vals"], [-10.0, -6.98, -5.22, -3.97, -3.01]):
            self.assertAlmostEqual(pair[0], pair[1], delta=0.01)
        
        for pair in zip(result["f0_vals"], [220.0, 330.0, 440.0, 330.0, 220.0]):
            self.assertAlmostEqual(pair[0], pair[1], delta=0.01)
                                
        self.assertAlmostEqual(result["spec_centroid"], 7.24, delta=0.01) # 7.32 is the correct
        
        for pair in zip(result["spec_slope"], [-0.0661, -0.0910, -0.1081]):            
            self.assertAlmostEqual(pair[0], pair[1], delta=0.001)
                                                     
        self.assertAlmostEqual(result["mean_spec_slope"], -0.088, delta=0.001)

                
        for pair in zip(result["spec_flux"], [0.000, 0.0347, 0.0347]):
            self.assertAlmostEqual(pair[0], pair[1], delta=0.001)
                
        self.assertAlmostEqual(result["mean_spec_flux"], 0.0232, 3)
                
        for pair in zip(result["spec_flat"], [0.717, 0.748, 0.751]):
            self.assertAlmostEqual(pair[0], pair[1], delta=0.001)
        
        self.assertAlmostEqual(result["mean_spec_flat"], 0.739, 2)
        

if __name__ == '__main__':
    unittest.main()
