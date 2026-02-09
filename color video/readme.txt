These programs where fully maid by 3 LLMs.

I gave prompt 1 to GPT-4, CLAUD, and DeepSeek. 
Then I copied all the code each LLM and then I feed each one prompt 2 and the code of all three LLMS
I then copied and paisted the code into PY files named for each LLM agent
I have yet to test how well they work

Prompt 1

ok so this may be a bit complicated but i will break down what i want. 
first of all bellow i have a program that breaks a video down into a chunk and then groups 
the pixels by cluster senters. this then sorts the cluster centers however I am skiptical 
about the constitencey of the way the sorting works as it does not seem to be consistent. 
here is what I want you to do. Have this program run off of one video. but cuts the first 
2 seconds off then does the clustering. here I want you to test the sorting find 6 methods 
of sorting and make a foulder labed sort types. in that folder start a foulder for each 
sort type. sort this fist chunk by each of those methods then render an image based on the 
results where the sorted order is a cool spectom. and drop the photo from the sorting method 
in the foulder. now for the next chunk instead of starting after the end of chunk 1, start 1 
fram after the start of chunk 1, such that chunk 1 and 2 share 255 frames, and so on untill 
you have compleated 30*8 chunks. then after all this the python code stickes all the photos 
togather to create a single video for each sorting method.

Prompt 2

ok now alnalisys the code produced by 2 other LLMs and your own code. figure out if somthing 
will break or error, and what you missed, and or what they missed, based on this analaysys, 
white a final copy, (also make it so that you specify the name of the one video file prossesd)