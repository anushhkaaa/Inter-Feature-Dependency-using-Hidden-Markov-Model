# Inter-feature dependencies of extracted driving features using Hidden-Markov-Model (HMM)
Road safety is a critical concern, and understanding the chain of events that lead to dangerous driving on the road is essential for enhancing safety measures. This project leverages an extensive dataset of on-board vehicular sensors like IMU, GPS and camera video data collected from vehicles in New York. The dataset contains the extracted 21 driving features of the ego vehicle that can ultimately lead to dangerous driving, such as braking, angle of swerving, number of cars, pedestrians, traffic lights, and more. Each of the 40 seconds videos were splitted into 5 seconds frames, from which the driving features were detected and classified using CNNs. These features are associated with their respective ground truths and time stamps.

## Project Highlights
- ### Hidden Markov Model (HMM) Analysis: We've employed Hidden Markov Models to obtain transition and emission probability matrices, allowing us to uncover inter-feature dependencies among the extracted driving features.
- ### Dangerous Region Identification: By identifying similar contexts over spatially co-located areas, we auto-annotate them as dangerous regions. This functionality enables us to predict alternate trajectories and enhance road safety.
- ### Applications: This work has broader applications and can be extended to devise safe path planning algorithms for autonomous vehicles and mobile robotic manipulators, contributing to the advancement of transportation and robotics technologies.

## Extracted Driving Features
Timestamp, pedestrian speed, weaving, swerving, sideslip, time of the day, road type, congestion, ground truths, jerking, relative speed and distance with other vehicles, turns, stops, and finally, number of cars, persons, buses, trucks and traffic lights

## Getting Started
1. Installation: Clone the repository and install the required dependencies.
2. Data: You will need access to a similar, extensive dataset. Due to data privacy and licensing issues, the dataset is not included in this repository.
3. HMM Analysis: The code for the preliminary stage involving Hidden Markov Models (HMM) analysis has been disclosed and can be found in the directory.
4. Run the Project: While the complete codebase of the next steps after using HMM cannot be publicly disclosed, you can explore the provided code, build upon it, and integrate it into your research or development efforts as per your requirement.

## License
This project is licensed under the MIT License -- see the LICENSE.md file for details.
