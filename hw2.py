import cv2
import matplotlib
import numpy as np
import math
from matplotlib import pyplot as plt


def findTextureSimilarities():
    differences = []
    histograms = []
    similarImgs = []
    for i in range(1, 41):
        if i < 10:
            path = ("Images/i0" + str(i) + ".jpg")
        else:
            path = ("Images/i" + str(i) + ".jpg")
        # print(path)
        histograms.append(getTextureVal(path))
    # print(len(histograms))
    # print(histograms)
    for r in range(0, 40):
        distances = []
        for c in range(0, 40):
            # if r != c:
            distances.append(L1Dist(histograms[r], histograms[c]))
        #    print(distances)
        similarImgs.append(top3(distances))
        differences.append(distances)
    index = 0
    pref = getPreferences()
    correct = 0
    for item in similarImgs:
        #    print(index, ": ", item, "\n")
        #print(item)
        for i in range(3):
        #    print(i)
        #    print("pref[index][item[i]-1]: ", pref[index][item[i]-1])
        #    print("pref[index][item[i]-1] != 0: ", pref[index][item[i] - 1]!= 0)

            if pref[index][item[i]-1] != 0:
                correct += 1
            #    print(correct)
        index += 1
    # print(len(similarImgs))
    outputHTML(similarImgs, "texture")
    print("Correct: ", correct)
    return differences


def findColorSimilarities():
    histograms = []
    similarImgs = []
    differences = []
    for i in range(1, 41):
        if i < 10:
            path = ("Images/i0" + str(i) + ".jpg")
        else:
            path = ("Images/i" + str(i) + ".jpg")
        # print(path)
        histograms.append(getColorHist(path))

    # print(distances)
    # print(histograms)

    for r in range(0, 40):
        distances = []
        for c in range(0, 40):
            # if r != c:
            distances.append(L1Dist(histograms[r], histograms[c]))

            # print("L1Distance: ",sum)
            # else:
            #    distances.append(0)
            # print(distances)
        # print("distances: ", distances)
        similarImgs.append(top3(distances))
        differences.append(distances)
    index = 0

    # print(len(similarImgs))
    pref = getPreferences()
    correct = 0
    for item in similarImgs:
        #    print(index, ": ", item, "\n")
        #print(item)
        for i in range(3):
        #    print(i)
            print("pref[index][item[i]-1]: ", pref[index][item[i] - 1])
            print("pref[index][item[i]-1] != 0: ", pref[index][item[i] - 1] != 0)

            if pref[index][item[i] - 1] != 0:
                correct += 1
        #        print(correct)
        index += 1
    print("Correct: ", correct)
    outputHTML(similarImgs, "color")
    return differences


def findShapeSimilarities():
    differences = []
    histograms = []
    similarImgs = []
    for i in range(1, 41):
        if i < 10:
            path = ("Images/i0" + str(i) + ".jpg")
        else:
            path = ("Images/i" + str(i) + ".jpg")
        # print(path)
        histograms.append(getShapeVal(path))

    for r in range(0, 40):
        distances = []
        for c in range(0, 40):
            bins = np.histogram_bin_edges(histograms[r])
            # print("bins: ", bins)
            if r != c:
                distances.append(histogramDiff(histograms[r], histograms[c]))
            else:
                distances.append(0)
        # print("distances: ", distances)
        similarImgs.append(top3(distances))
        differences.append(distances)
    index = 0
    pref = getPreferences()
    correct = 0
    for item in similarImgs:
        #    print(index, ": ", item, "\n")
        # print(item)
        for i in range(3):
            #    print(i)
            #    print("pref[index][item[i]-1]: ", pref[index][item[i] - 1])
            #    print("pref[index][item[i]-1] != 0: ", pref[index][item[i] - 1] != 0)

            if pref[index][item[i] - 1] != 0:
                correct += 1
        #        print(correct)
        index += 1
    print("Correct: ", correct)
    # print(len(similarImgs))
    outputHTML(similarImgs, "shape")
    return differences


def histogramDiff(h1, h2):
    sm = 0
    for i in range(len(h1)):
        sm += min(h1[i], h2[i])
    # print("sm: ", 1 - sm / (60 * 89))
    return 1 - sm[0] / (60 * 89)


def outputHTML(similarImgs, type):
    file = open(type + "Eval.html", "w")
    file.write("<html><body><table>")
    textScores = crowdScore()
    # print("textScores: ", textScores)
    scores = []
    index = 1
    for i in similarImgs:

        file.write("\n<tr>")
        if index < 10:
            file.write('\n<td><img src="Images/i0' + str(index) + '.jpg"width = "44px" height = "20px"><br>0' + str(
                index) + '</td>')
        else:
            file.write('\n<td><img src="Images/i' + str(index) + '.jpg"width = "44px" height = "20px"><br>' + str(
                index) + '</td>')
        scoreSum = 0
        pos = 0
        for item in i:
            # print("item: ", item)
            scoreSum += textScores[index - 1][i[pos] - 1]
            # print("scoreSum: ", scoreSum)
            if item < 10:
                file.write('\n<td><img src="Images/i0' + str(item) + '.jpg" width = "44px" height = "20px"><br>0' + str(
                    item) + ', ' + str(
                    textScores[index - 1][i[pos] - 1]) + '</td>')
            else:
                file.write('\n<td><img src="Images/i' + str(item) + '.jpg"width = "44px" height = "20px"><br>' + str(
                    item) + ', ' + str(
                    textScores[index - 1][i[pos] - 1]) + '</td>')
            pos += 1
        file.write("\n<td>Score: " + str(scoreSum) + "</td>")
        scores.append(
            textScores[index - 1][i[0] - 1] + textScores[index - 1][i[1] - 1] + textScores[index - 1][i[2] - 1])
        index += 1
        file.write("\n</td>")
    sum = 0
    for item in scores:
        sum += item
    file.write("Total score: " + str(sum))
    file.write("</html></body></table>")
    file.close()


def crowdScore():
    crowd = open("Crowd.txt")
    lines = []
    for i in range(0, 40):
        line = crowd.readline()
        # print(line)
        newLine = line.split(" ")
        # print(newLine)
        index = 0
        for num in range(0, len(newLine)):
            if newLine[num] == '':
                index += 1
        for num in range(0, index):
            newLine.remove("")

        newLine.remove("\n")

        index = 0
        for item in newLine:
            newLine[index] = int(item)
            index += 1
        lines.append(newLine)
    # print(lines)
    return lines


def top3(array):
    # print(array)
    val1 = 1
    index1 = 0
    val2 = 1
    index2 = 0
    val3 = 1
    index3 = 0
    index = 0

    worst = 0
    worstIndex = 0

    for item in array:
        if item != 0:
            if item < val1:
                val3 = val2
                index3 = index2
                val2 = val1
                index2 = index1
                val1 = item
                index1 = index
            elif item < val2:
                val3 = val2
                index3 = index2
                val2 = item
                index2 = index
            elif item < val3:
                val3 = item
                index3 = index
            elif item > worst:
                worst = item
                worstIndex = index
        index += 1
    return index1 + 1, index2 + 1, index3 + 1, worstIndex + 1


def L1Dist(im1, im2):
    sum = 0.0
    # norm = abs(im1 - im2)
    norm = im1 - im2
    nsum = np.linalg.norm(norm) / (2 * 60 * 89)
    nsquared = np.square(norm)
    sum = np.sum(nsquared)
    # for item in nsquared:
    #    for value in item:
    #        try:
    #            for a in value:
    #                sum += a
    #        except:
    #            sum += value
    sum = math.sqrt(sum)
    return sum / (2 * 60 * 89)
    # return nsum


def getColorHist(path):
    img = cv2.imread(path)
    ppm = cv2.imread("Images/i01.ppm")
    n_channels = img.shape[2]
    channels = list(range(n_channels))
    ranges = [0, 256] * n_channels
    hist = cv2.calcHist([img], channels, None, [5, 8, 4], ranges)

    # cv2.imshow(img)
    # print(img.shape)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return hist


def getTextureVal(path):
    img = cv2.imread(path, 0)

    '''I tested both the full laplacian numpy array and the absolute value adjusted version, the absolute value
    version did not show enough of a difference between images to  merit use, therefore I simply went with the full
    values given my transforming an image using the laplacian transformation.'''
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    '''Though I recognize 10000 bins is a lot, I chose this number as it gave me the highest total score out of all the 
    other bin values I tried, essentially testing through trial and error. '''
    hist = cv2.calcHist(laplacian.astype('float32'), [0], None, [10000], [-5400, 5400])

    return hist


def getShapeVal(path):
    img = cv2.imread(path, 0)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    hist = cv2.calcHist([img], [0], thresh1, [64], [0, 256])

    return hist


def getPreferences():
    crowd = open("MyPreferences.txt")
    lines = []
    for i in range(0, 40):
        line = crowd.readline()
        # print(line)
        newLine = line.split(" ")
        # print(newLine)
        index = 0
        for num in range(0, len(newLine)):
            if newLine[num] == '':
                index += 1
        for num in range(0, index):
            newLine.remove("")

        #newLine.remove("\n")

        index = 0
        for item in newLine:
            newLine[index] = int(item)
            index += 1
        lines.append(newLine)
    # print(lines)
    return lines


def evaluateImages():
    color = np.array(findColorSimilarities())
    texture = np.array(findTextureSimilarities())
    shape = np.array(findShapeSimilarities())
    # print(color.shape)
    # print(texture.shape)
    # print(shape.shape)
    # print(color[0][0])
    total =[]

    index1 = 0
    for i in color:
        index2 = 0
        values = []
        for j in i:
            values.append(.5*color[index1][index2] + .45*texture[index1][index2] + .05*shape[index1][index2])
            index2 += 1
        total.append(values)
        index1+=1
    # print("total: ", total)

    similarImgs = []
    for item in total:
        similarImgs.append(top3(item))

    index = 1
    #for item in similarImgs:
    #    print(index, " : ", item)
    index = 0
    pref = getPreferences()
    correct = 0
    for item in similarImgs:
        #    print(index, ": ", item, "\n")
        # print(item)
        for i in range(3):
            #    print(i)
            #    print("pref[index][item[i]-1]: ", pref[index][item[i] - 1])
            #    print("pref[index][item[i]-1] != 0: ", pref[index][item[i] - 1] != 0)

            if pref[index][item[i] - 1] != 0:
                correct += 1
        #        print(correct)
        index += 1
    print("Correct: ", correct)
    outputHTML(similarImgs, "overall")




if __name__ == '__main__':
    # hist = getColorHist("Images/i01.jpg")
    # print(np.sum(hist))
    # determineScore()
    # findColorSimilarities()
    # getTextureVal("Images/i01.jpg")
    # findTextureSimilarities()
    # getShapeVal("Images/i24.jpg")
    # findShapeSimilarities()
    print(getPreferences())
    evaluateImages()

    crowd = crowdScore()
    possible = []
    sum = 0
    for item in crowd:
        item.sort(reverse=True)
        sum += item[0] + item[1] + item[2]

    print(sum)
