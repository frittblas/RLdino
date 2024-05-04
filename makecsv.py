import csv

def process_file(filename):
    # Read the file and filter out lines starting with '1'
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    filtered_lines = [line for line in lines if not line.startswith('1')]

    # Extract scores from filtered lines and round them to the nearest integer
    scores = []
    for line in filtered_lines:
        if "You Loose! Points:" in line:
            score = float(line.split("You Loose! Points: ")[1].strip())
            rounded_score = round(score)
            scores.append({"You Loose! Points": rounded_score})
        elif "You Win! Points:" in line:
            score = float(line.split("You Win! Points: ")[1].strip())
            rounded_score = round(score)
            scores.append({"You Win! Points": rounded_score})

    # Save scores to CSV file
    csv_filename = "scores_hans_400.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['You Loose! Points', 'You Win! Points']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for score in scores:
            writer.writerow(score)

    print(f"Scores saved to {csv_filename}")


def main():
    process_file("output_hans_4.txt")

if __name__ == "__main__":
    main()
    