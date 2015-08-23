Implementation of linear regression , gradient descent algorithm. The implementation follows Andrew Ng's coursera course on Machine learning. 

To run the code: 

	Train your data by running:

		python optimizeGD.py --train data_file.txt

	(to run the sample given)
		
		python optimizeGD.py -t ex1data2.txt

	The format of the data file is explained below. The program computes the optimization parameters `theta' and saves them in 
	a text file 'params.txt'. 

	Training data format: 

		Suppose the feature vector is X0=[x0_0,x0_1,x0_2,x0_3] and the correspoding output value is y0. The 
		data to be store in the file is then stored as (assuming there as N+1 traning samples with us)
		-------------	
		
		x0_0, x0_1, x0_2, x0_3, y0
		x1_0, x1_1, x1_2, x1_3, y1
		.
		.
		xN_0,xN_1, xN_2, xN_3, yN

		-------
		e.g 
		-----------
		1, 3, 4, 5, 6
		2, 1, 6, 7, 9
		2, 5, 1, 10, 11
		--------
		where the input matrix is 
		   __	        _	
		   | 1, 3, 4, 5  |	
		X= | 2, 1, 6, 7  |
		   |_2, 5, 1, 10_|	
		    
		Each row is a traning sample and each column represents a feature 
                   _   _
		  |  6  |
		Y=|  9  |
		  |_ 11_|

Prediction

	Once the data is trained the predicted value for a given input can be obtained as follows. 
	
	1) Input value using command line (assuming the input is a vector of length 2;  [val1 val2])
		
		python optimizeGD.py --predict val1 val2 
	
	(assuming you have run the above traning example try)  
		
		python optimizeGD.py -p 3 2

	2) Input value using a file 

		python optimizeGD.py --input input_file.txt 

	
	(assuming you have run the above traning example try) 
		
		python optimizeGD.py -i test_data2.txt

	Input data format:
		
		Suppose you want to predict the values for the given set of feature values. 
		   __	        _	
		   | 3, 2, 1, 2  |	
		X= | 0, 2, 4, 3  |
		   |_1, 0, 7, 10_|	
		    
		The vectors are stored in a file as give below
		-------------------

		    3, 2, 1, 2  	
		    0, 2, 4, 3  
		    1, 0, 7, 10	
		    
		Check out the sample test data for test_data2.txt for feature vectors of lenght 2		

