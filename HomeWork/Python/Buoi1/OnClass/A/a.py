# Module : statistics
import statistics
listNumber = [2, 2, 5, 6, 9, 30]
mean_value = statistics.mean(listNumber)
median_value = statistics.median(listNumber)
mode_value = statistics.mode(listNumber)
std_value = statistics.stdev(listNumber)
print(mean_value, median_value, mode_value, std_value)
