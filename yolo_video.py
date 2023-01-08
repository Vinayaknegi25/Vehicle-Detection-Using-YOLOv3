import numpy as np
import imutils
import time
from scipy import spatial
import cv2
from input_retrieval import *
list_of_vehicles = ["bicycle","car","motorbike","bus","truck"]
FRAMES_BEFORE_CURRENT = 10  
inputWidth, inputHeight = 416, 416

LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath,\
	preDefinedConfidence, preDefinedThreshold= parseCommandLineArguments()

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

# parameters: Frame on which the count is displayed, the count number of vehicles 
def displayVehicleCount(frame, vehicle_count):
	cv2.putText(
		frame, 
		'Number Of Vehicles: ' + str(vehicle_count), 
		(20, 40), 
		cv2.FONT_HERSHEY_SIMPLEX, 
		1.8,
		(0, 0, 0),
		2, #Thickness
		cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)

def displayFPS(start_time, num_frames):
    current_time = int(time.time())
    if(current_time > start_time):
        os.system('clear')
        print("FPS:", num_frames)
        num_frames = 0
        start_time = current_time
    return start_time, num_frames

def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
	if len(idxs) > 0:
		# loop over the indices we are keeping
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def initializeVideoWriter(video_width, video_height, videoStream):
	sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
	# initialize our video writer
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps,(video_width, video_height), True)

 # Takes all the vehicular detections of the previous frames, the coordinates of the box of previous detections
def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
	centerX, centerY, width, height = current_box
	dist = np.inf
	# Iterating through all the k-dimensional
	for i in range(FRAMES_BEFORE_CURRENT):
		coordinate_list = list(previous_frame_detections[i].keys())
		if len(coordinate_list) == 0: # if no detections in the previous frame
			continue
		temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
		if (temp_dist < dist):
			dist = temp_dist
			frame_num = i
			coord = coordinate_list[index[0]]

	if (dist > (max(width, height)/2)):
		return False

	# Keeping the vehicle ID constant
	current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
	return True

def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame):
	current_detections = {}
	if len(idxs) > 0:
		for i in idxs.flatten():
			# the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			centerX = x + (w//2)
			centerY = y+ (h//2)

			if (LABELS[classIDs[i]] in list_of_vehicles):
				current_detections[(centerX, centerY)] = vehicle_count 
				if (not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)):
					vehicle_count += 1

				ID = current_detections.get((centerX, centerY))
				# If there are two detections having the same ID due to being too close, 
				if (list(current_detections.values()).count(ID) > 1):
					current_detections[(centerX, centerY)] = vehicle_count
					vehicle_count += 1 

				cv2.putText(frame, str(ID), (centerX, centerY),\
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)

	return vehicle_count, current_detections

#Program Starts here
print("Loading YOLO from File!!")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

videoStream = cv2.VideoCapture(inputVideoPath)
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

x1_line = 0
y1_line = video_height//2
x2_line = video_width
y2_line = video_height//2

#Initialization
previous_frame_detections = [{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]
num_frames, vehicle_count = 0, 0
writer = initializeVideoWriter(video_width, video_height, videoStream) #to get fps from the video
start_time = int(time.time())
# loop over frames from the video file stream
while True:
	print("----------------------------NEW FRAME-----------------------------------------")
	num_frames+= 1
	print("FRAME:\t", num_frames)
	boxes, confidences, classIDs = [], [], [] 
	vehicle_crossed_line_flag = False 

	start_time, num_frames = displayFPS(start_time, num_frames)
	(grabbed, frame) = videoStream.read()

	if not grabbed:
		break

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight),swapRB=True, crop=False)

	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	for output in layerOutputs:
		for i, detection in enumerate(output): 
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > preDefinedConfidence:
				box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
				(centerX, centerY, width, height) = box.astype("int")

				# to calculete the top and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,
		preDefinedThreshold)

	drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

	vehicle_count, current_detections = count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame)

	displayVehicleCount(frame, vehicle_count)

	writer.write(frame)

	cv2.imshow('Frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('v'):
		break	
	
	previous_frame_detections.pop(0) #Removing the first frame from the list
	previous_frame_detections.append(current_detections)

print("Cleaning up...")
writer.release()
videoStream.release()
