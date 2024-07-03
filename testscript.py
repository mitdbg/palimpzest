from palimpzest.dataclasses import GenerationStats, RecordOpStats

x = GenerationStats()
y = GenerationStats(total_input_tokens=3)

print(y,"\n")
z = x+y
print(z,"\n")
x+=y
print(x,"\n")
total_generation_stats = x+y

r = RecordOpStats(
    record_uuid = "1",
    record_parent_uuid = "2",
    op_id = "3",
    op_name = "4",
    time_per_record = 5,
    cost_per_record = 6,
    record_state = {}
    ,  **total_generation_stats.__dict__)

print(r)