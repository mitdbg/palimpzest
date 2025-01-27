import csv
import os

email_files = os.listdir('testdata/enron-tiny')
all_data = []

for file_path in email_files:
    # if file_path != "9.":
    #     continue

    with open(f'testdata/enron-tiny/{file_path}') as file:
        content = file.read()

        # Split lines and extract key-value pairs
        lines = content.splitlines()
        data = {}
        data['filename'] = file_path
        for line in lines:
            if ': ' in line:
                key, value = line.split(': ', 1)
                data[key.strip()] = value.strip()
            else:
                if 'body' in data:
                    data['body'] += f" {line.strip()}"
                else:
                    data['body'] = line.strip()

        all_data.append(data)



# Define headers from all collected data
headers = ['filename', 'Message-ID', 'Date', 'From', 'To', 'Subject',
           'Mime-Version', 'Content-Type', 'Content-Transfer-Encoding',
           'X-From', 'X-To', 'X-cc', 'X-bcc', 'X-Folder', 'X-Origin',
           'X-FileName', 'body']


# all_headers = set()
# for data_item in all_data:
#     all_headers.update(data_item.keys())


with open('testdata/enron-tiny.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)  # Write the header

    for data in all_data:
        writer.writerow([data.get(header, '') for header in headers])  # Write each email's values


print(f"CSV file created: testdata/enron-tiny.csv with {len(all_data)} emails processed")
print("CSV file created: testdata/enron-tiny.csv")
