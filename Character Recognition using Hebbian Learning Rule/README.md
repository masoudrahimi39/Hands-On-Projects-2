### project description

one of the applications of Hebb's learning law is letter recognition. We want to design a network that by giving 7*9 letters can produce their 3*5 output. To define these letters in Python, assign a -1 to the dots and a +1 to the #'s.
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/1a80da74-f5c3-4eb9-ab52-7f95573c42a0)

#### answer to below questiona:
1) Will this network be able to bring all the inputs to the desired output?


**Next, we want to measure the resistance of this network against noise (replacing the numbers +1 and -1) and information loss (replacing zero instead of +1 and -1).**

2) In the first step, randomly add 20 and 40% noise to the network. What percentage of the time will the network succeed in correctly identifying the output? (Hint: to calculate your code
Run it several times and calculate the percentage of times the network was able to correctly reach all three outputs.)

3) In the next step, remove 20 and 40% of the image information (instead of +1 and -1, put zero) in what percentage of times will the network succeed in recognizing the output correctly?

4) The resistance of the network is greater against which one? Noise or data loss? show.

5) State the maximum resistance of the network against noise (hint: repeat the tests with different percentages until you get a reliable result)
