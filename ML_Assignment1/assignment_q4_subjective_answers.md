# Question - 4 
```
In the implementation done in the code experiments.py :
M = No.of Features in the dataset  
N = No. of samples   

For every possible case we have run the code 3 times to calculate the average time and the standard deviation of the same. For every possible case i.e. Discrete Input Discrete Output, Discrete Input Real output, Real Input Real Output, Real Input Discrete Output we define a (N,M) dataset where N ranges from (2,4) and M ranges from (10,40) with finite step being 10.

For each plot we have plotted the mean and standard deviation corresponding to each case. On x axis we have plotted the specific sequence of dataset i.e. (N,M) values. Since total N values range from 2 to 4 (i.e 3 options) and M values range from 10 to 40 and finite step being 10 (i.e 4 options). Thus, the x axis range from 0 to 12 (i.e 3*4 for all options). 


Output from the code for the mean and the standard deviation corresponding to each case:

sample: 2 x 10, Mean fit time: 0.04248801867167155, std: 0.007674637650730873
sample: 2 x 20, Mean fit time: 0.12186876932779948, std: 0.00974817217683921
sample: 2 x 30, Mean fit time: 0.16128055254618326, std: 0.003858637488599108
sample: 2 x 40, Mean fit time: 0.2582403818766276, std: 0.0010443875413942236
sample: 3 x 10, Mean fit time: 0.05077830950419108, std: 0.007930134652664551
sample: 3 x 20, Mean fit time: 0.12388412157694499, std: 0.010871699791216804
sample: 3 x 30, Mean fit time: 0.23841118812561035, std: 0.006054837503164542
sample: 3 x 40, Mean fit time: 0.37531669934590656, std: 0.010127859766360001
sample: 4 x 10, Mean fit time: 0.05985609690348307, std: 0.003274046376707533
sample: 4 x 20, Mean fit time: 0.15167236328125, std: 0.0037637393087293232
sample: 4 x 30, Mean fit time: 0.3204202651977539, std: 0.014760952162556958
sample: 4 x 40, Mean fit time: 0.4900677998860677, std: 0.012610072646917301
sample: 2 x 10, Mean predict time: 0.0027391910552978516, the standard deviation is 0.0038738011403332924
sample: 2 x 20, Mean predict time: 0.0, the standard deviation is 0.0
sample: 2 x 30, Mean predict time: 0.00864712397257487, the standard deviation is 0.0004808122067391525
sample: 2 x 40, Mean predict time: 0.002828518549601237, the standard deviation is 0.004000129294269945
sample: 3 x 10, Mean predict time: 0.0, the standard deviation is 0.0
sample: 3 x 20, Mean predict time: 0.0026682217915852866, the standard deviation is 0.00377343544507935
sample: 3 x 30, Mean predict time: 0.0, the standard deviation is 0.0
sample: 3 x 40, Mean predict time: 0.0020035107930501304, the standard deviation is 0.002833392135892369
sample: 4 x 10, Mean predict time: 0.0, the standard deviation is 0.0
sample: 4 x 20, Mean predict time: 0.0, the standard deviation is 0.0
sample: 4 x 30, Mean predict time: 0.0012733141581217449, the standard deviation is 0.0018007381515774512
sample: 4 x 40, Mean predict time: 0.002793550491333008, the standard deviation is 0.003950676992017163
Finished RIRO!
Started RIDO!
2 10
2 20
2 30
2 40
3 10
3 20
3 30
3 40
4 10
4 20
4 30
4 40


sample: 2 x 10, case - 2, Mean fit time: 0.09252023696899414, std: 0.0033430859381493572
sample: 2 x 20, case - 2, Mean fit time: 6.048991759618123, std: 0.05505372182300552
sample: 2 x 30, case - 2, Mean fit time: 2.337550640106201, std: 0.035432030065682164
sample: 2 x 40, case - 2, Mean fit time: 90.50851504007976, std: 2.2304467220319
sample: 3 x 10, case - 2, Mean fit time: 0.5355041821797689, std: 0.053587872591643826
sample: 3 x 20, case - 2, Mean fit time: 37.46234806378683, std: 1.5515622355434373
sample: 3 x 30, case - 2, Mean fit time: 26.4102783203125, std: 0.7155419498543975
sample: 3 x 40, case - 2, Mean fit time: 94.19592467943828, std: 0.11247219181370638
sample: 4 x 10, case - 2, Mean fit time: 0.6734418074289957, std: 0.021994585111728842
sample: 4 x 20, case - 2, Mean fit time: 80.84851241111755, std: 2.4397479302067704
sample: 4 x 30, case - 2, Mean fit time: 219.2063185373942, std: 4.919742885079955
sample: 4 x 40, case - 2, Mean fit time: 674.6803110440572, std: 197.39718920440527


sample: 2 x 10, case - 2, Mean predict time: 0.0, std: 0.0
sample: 2 x 20, case - 2, Mean predict time: 0.0026841958363850913, std: 0.0037960261558811887
sample: 2 x 30, case - 2, Mean predict time: 0.0, std: 0.0
sample: 2 x 40, case - 2, Mean predict time: 0.0028122266133626304, std: 0.003977089017083989
sample: 3 x 10, case - 2, Mean predict time: 0.0, std: 0.0
sample: 3 x 20, case - 2, Mean predict time: 0.00341796875, std: 0.004833737762017415
sample: 3 x 30, case - 2, Mean predict time: 0.0003559589385986328, std: 0.0005034019586141183
sample: 3 x 40, case - 2, Mean predict time: 0.0026076634724934897, std: 0.0036877930489052128
sample: 4 x 10, case - 2, Mean predict time: 0.0, std: 0.0
sample: 4 x 20, case - 2, Mean predict time: 0.0, std: 0.0
sample: 4 x 30, case - 2, Mean predict time: 0.0, std: 0.0
sample: 4 x 40, case - 2, Mean predict time: 0.0013345082600911458, std: 0.0009436400209752985
Finsihed RIDO!
Started DIRO!
2 10
2 20
2 30
2 40
3 10
3 20
3 30
3 40
4 10
4 20
4 30
4 40


sample: 2 x 10, case - 3, Mean fit time: 0.03329348564147949, std: 0.0041819432479774255
sample: 2 x 20, case - 3, Mean fit time: 0.05182329813639323, std: 0.005933356089651475
sample: 2 x 30, case - 3, Mean fit time: 0.07306210199991862, std: 0.005570441344343526
sample: 2 x 40, case - 3, Mean fit time: 0.08913644154866536, std: 0.016065552950736733
sample: 3 x 10, case - 3, Mean fit time: 0.030316750208536785, std: 0.002036768280459092
sample: 3 x 20, case - 3, Mean fit time: 0.062388340632120766, std: 0.0006051890253544793
sample: 3 x 30, case - 3, Mean fit time: 0.09017594655354817, std: 0.0021859935818214784
sample: 3 x 40, case - 3, Mean fit time: 0.11403393745422363, std: 0.0030043782037375644
sample: 4 x 10, case - 3, Mean fit time: 0.028357505798339844, std: 0.0018373785715641545
sample: 4 x 20, case - 3, Mean fit time: 0.05401142438252767, std: 0.0029029793344343196
sample: 4 x 30, case - 3, Mean fit time: 0.08362317085266113, std: 0.0005593160972811131
sample: 4 x 40, case - 3, Mean fit time: 0.11332496007283528, std: 0.007017602486935363


sample: 2 x 10, case -3, Mean predict time: 0.003609895706176758, std: 0.0007397621253979168
sample: 2 x 20, case -3, Mean predict time: 0.007459322611490886, std: 0.0009450633060064294
sample: 2 x 30, case -3, Mean predict time: 0.010531981786092123, std: 0.0004020025868409599
sample: 2 x 40, case -3, Mean predict time: 0.013958930969238281, std: 0.0006851912978623996
sample: 3 x 10, case -3, Mean predict time: 0.004076560338338216, std: 5.694506299774447e-05
sample: 3 x 20, case -3, Mean predict time: 0.005710045496622722, std: 0.0005062254853563488
sample: 3 x 30, case -3, Mean predict time: 0.009763081868489584, std: 0.0005173784002052923
sample: 3 x 40, case -3, Mean predict time: 0.013190746307373047, std: 0.001007277235807516
sample: 4 x 10, case -3, Mean predict time: 0.0023719469706217446, std: 0.0004427669600195485
sample: 4 x 20, case -3, Mean predict time: 0.007005532582600911, std: 0.0008124747665115406
sample: 4 x 30, case -3, Mean predict time: 0.009178082148234049, std: 0.0010340017489062906
sample: 4 x 40, case -3, Mean predict time: 0.013796011606852213, std: 0.0005801676721904523
Finsihed DIRO!
Starting DIDO!
2 10
2 20
2 30
2 40
3 10
3 20
3 30
3 40
4 10
4 20
4 30
4 40
sample: 2 x 10, case - 3, Mean predict time: 1.4503850142161052, std: 0.03132656223361516
sample: 2 x 20, case - 3, Mean predict time: 2.5737148920694985, std: 0.04278722616069402
sample: 2 x 30, case - 3, Mean predict time: 3.546570301055908, std: 0.06938076173178313
sample: 2 x 40, case - 3, Mean predict time: 5.0173139572143555, std: 0.12419788178344088
sample: 3 x 10, case - 3, Mean predict time: 0.5048638184865316, std: 0.029481068779324604
sample: 3 x 20, case - 3, Mean predict time: 1.4398337999979656, std: 0.03199151593185423
sample: 3 x 30, case - 3, Mean predict time: 3.3332438468933105, std: 0.0450074021589394
sample: 3 x 40, case - 3, Mean predict time: 4.7881457805633545, std: 0.15284079492269279
sample: 4 x 10, case - 3, Mean predict time: 0.4776319662729899, std: 0.017978692095764703
sample: 4 x 20, case - 3, Mean predict time: 1.2732176780700684, std: 0.028664749325088598
sample: 4 x 30, case - 3, Mean predict time: 2.6625062624613443, std: 0.1404499038358498
sample: 4 x 40, case - 3, Mean predict time: 3.1804657777150473, std: 0.05063041211914015


sample: 2 x 10, case - 3, Mean predict time: 0.003000656763712565, std: 5.296733443880601e-06
sample: 2 x 20, case - 3, Mean predict time: 0.0063822269439697266, std: 0.00044957159624625496
sample: 2 x 30, case - 3, Mean predict time: 0.009826421737670898, std: 0.0011637666328377448
sample: 2 x 40, case - 3, Mean predict time: 0.009969075520833334, std: 0.007054967461380355
sample: 3 x 10, case - 3, Mean predict time: 0.0010103384653727214, std: 0.001428834360317322
sample: 3 x 20, case - 3, Mean predict time: 0.01211078961690267, std: 0.008618199588498823
sample: 3 x 30, case - 3, Mean predict time: 0.008503595987955729, std: 0.006508644207783234
sample: 3 x 40, case - 3, Mean predict time: 0.014252821604410807, std: 0.0019569799884030154
sample: 4 x 10, case - 3, Mean predict time: 0.0, std: 0.0
sample: 4 x 20, case - 3, Mean predict time: 0.0026569366455078125, std: 0.0037574758384432247
sample: 4 x 30, case - 3, Mean predict time: 0.010939995447794596, std: 0.0007095495092487164
sample: 4 x 40, case - 3, Mean predict time: 0.013544082641601562, std: 0.0004901228822328386
YAAY! Finsihed DIDO!



From the output, it can be seen that fitting the model to real input data takes more time because doing so involves looking at all potential splits in the feature values. This is due to the fact that for actual input, we must determine the best split for each row in the column. In contrast, when the inputs are discrete, we may easily determine the split by grouping labels that are similar.

For a balanced tree, the depth of a tree is O(logN). Here N is no. of samples. 
In the worst case scenarios, the depth can go upto O(N). Thus theoritically the worst case time complexity would be O(NMd) where d is the depth of the tree. Decision tree algorithm is a greedy algorithm as it splits locally with uncertainity.   

Our Experimental observations show that the time complexity is O(log(N)*M) .
```

## Plots 

### Real input Real output

![fit](https://github.com/ES654/assignment-1-hm/tree/master/plots/fit_time_riro.png)
![predict](https://github.com/ES654/assignment-1-hm/tree/master/plots/predict_time_riro.png)  

### Real input Discrete output

![fit](https://github.com/ES654/assignment-1-hm/tree/master/plots/fit_time_rido.png)
![predict](https://github.com/ES654/assignment-1-hm/tree/master/plots/predict_time_rido.png)

### Discrete input Real output

![fit](https://github.com/ES654/assignment-1-hm/tree/master/plots/fit_time_diro.png)
![predict](https://github.com/ES654/assignment-1-hm/tree/master/plots/predict_time_diro.png)

### Discrete input Discrete output

![fit](https://github.com/ES654/assignment-1-hm/tree/master/plots/fit_time_dido.png)
![predict](https://github.com/ES654/assignment-1-hm/tree/master/plots/predict_time_dido.png)

