import numpy as np
import matplotlib.pyplot as plt

#One job
time_30 = [0.010030100005678833, 0.00932090007700026, 0.008952200063504279]
time_500 = [0.013635499984957278, 0.01420269999653101, 0.01379090000409633]
time_1000 = [0.024013299960643053, 0.021817200002260506, 0.021967199980281293]

time_30_avg_njobs1 = np.mean(time_30)
time_500_avg_njobs1  = np.mean(time_500)
time_1000_avg_njobs1  = np.mean(time_1000)

#Many jobs
time_30 = [0.014879100024700165 , 0.014346800046041608, 0.013564500026404858]
time_500 = [0.015721999923698604, 0.014661399996839464, 0.01535819994751364]
time_1000 = [0.02047839993610978, 0.019824299961328506, 0.019313299912028015]

time_30_avg_njobsmany  = np.mean(time_30)
time_500_avg_njobsmany  = np.mean(time_500)
time_1000_avg_njobsmany  = np.mean(time_1000)

jobs1 = [time_30_avg_njobs1, time_500_avg_njobs1, time_1000_avg_njobs1]
jobsmany = [time_30_avg_njobsmany, time_500_avg_njobsmany, time_1000_avg_njobsmany]

plt.plot([30,500,1000],jobs1, label = "OneJob")
plt.plot([30,500,1000],jobsmany, label = "ManyJobs")
plt.xlabel('Num_Datapoints')
plt.ylabel('Time(s)')
plt.title('One job vs many')
plt.legend()
plt.savefig('figures/q4_timing_analysis.png')
# plt.show()