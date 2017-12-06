// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sfa.SFAWordsTest;
import sfa.timeseries.MultiDimTimeSeries;
import sfa.timeseries.MultiDimTimeSeriesLoader;

import java.io.File;
import java.io.IOException;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertNotNull;


@RunWith(JUnit4.class)
public abstract class AbstractClassifierMDTest {

    private static final double DELTA = 0.05;
    protected static final File DATASETS_DIRECTORY = new File(
            AbstractClassifierMDTest.class.getClassLoader().getResource("datasets/").getFile());

    @Test
    public void testClassificationOnUCRData() {
        // the relative path to the datasetsArray
        ClassLoader classLoader = SFAWordsTest.class.getClassLoader();
        for (DataSet dataSet : getDataSets()) {
            MDClassifier classifier = trainClassifier(dataSet);
            assertNotNull(classifier);
        }
    }


    public void testSave() throws IOException {
        DataSet dataSet = this.getDataSets().get(0);
        testSaveLoadGivesEqualTestResults(dataSet);
    }

    private void testSaveLoadGivesEqualTestResults(DataSet dataSet) throws IOException {
        MDClassifier classifier=trainClassifier(dataSet);
        File file=createTempClassifierFile();
        classifier.save(file);
        Classifier loadedClassifier = Classifier.load(file);
        Assert.assertNotNull(loadedClassifier);
        checkEqualResultsOfClassifiers(dataSet, classifier, loadedClassifier);
    }

    private void checkEqualResultsOfClassifiers(DataSet dataSet, MDClassifier classifier, MDClassifier loadedClassifier) {
        MultiDimTimeSeries[] samples = MultiDimTimeSeriesLoader.loadDataset(getFirstTrainFile(dataSet));
        MDClassifier.Predictions loadedScore = loadedClassifier.score(samples);
        MDClassifier.Predictions score = classifier.score(samples);

        Assert.assertArrayEquals(loadedScore.labels, score.labels);
        Assert.assertEquals(loadedScore.correct.get(), score.correct.get());
    }

    private File getFirstTrainFile(DataSet dataset) {
        return getTrainFiles(dataset)[0];
    }

    private File createTempClassifierFile() throws IOException {
        File tmpFile = File.createTempFile("classifier", "class");
        tmpFile.deleteOnExit();
        return tmpFile;
    }

    protected static final class DataSet {
        public DataSet(String name, double trainingAccuracy, double testingAccuracy) {
            this.name = name;
            this.trainingAccuracy = trainingAccuracy;
            this.testingAccuracy = testingAccuracy;
        }

        String name;
        double trainingAccuracy, testingAccuracy;
    }

    protected MDClassifier trainClassifier(DataSet dataSet) {
        File[] trainFiles = getTrainFiles(dataSet);

        MDClassifier classifier = null;
        for (File train : trainFiles) {
            File test = new File(train.getAbsolutePath().replaceFirst("TRAIN", "TEST"));

            if (!test.exists()) {
                System.err.println("File " + test.getName() + " does not exist");
                test = null;
            }

            MDClassifier.DEBUG = false;

            // Load the train/test splits
            MultiDimTimeSeries[] testSamples = MultiDimTimeSeriesLoader.loadDataset(test);
            MultiDimTimeSeries[] trainSamples = MultiDimTimeSeriesLoader.loadDataset(train);

            classifier = initClassifier();
            MDClassifier.Score scoreW = classifier.eval(trainSamples, testSamples);
            assertEquals("testing result of " +
                    dataSet.name+" does NOT match",
                    dataSet.testingAccuracy,
                    scoreW.getTestingAccuracy(),
                    DELTA);
            assertEquals("training result of "+dataSet.name+" does NOT match",
                    dataSet.trainingAccuracy,
                    scoreW.getTrainingAccuracy(),
                    DELTA);
            System.out.println(scoreW.toString());

        }
        return classifier;
    }

    protected File[] getTrainFiles(DataSet dataSet) {
        File dataSetDirectory = new File(DATASETS_DIRECTORY.getAbsolutePath()+"/"+dataSet.name);
        return getTrainFilesFromDir(dataSetDirectory);
    }

    private File[] getTrainFilesFromDir(File dataSetDirectory) {
        return dataSetDirectory.listFiles(pathname -> pathname.getName().toUpperCase().endsWith("TRAIN"));
    }

    protected abstract List<DataSet> getDataSets();


    protected abstract MDClassifier initClassifier();
}
