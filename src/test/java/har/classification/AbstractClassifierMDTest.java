package har.classification;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sfa.SFAWordsTest;
import har.timeseries.MultiDimTimeSeries;
import har.timeseries.MultiDimTimeSeriesLoader;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertNotNull;


@RunWith(JUnit4.class)
public abstract class AbstractClassifierMDTest {
    protected static final String TEST_FILE = "TEST";
    protected static final String TRAIN_FILE = "TRAIN";
    private static final double DELTA = 0.05;
    protected static final File DATASETS_DIRECTORY = new File(
            AbstractClassifierMDTest.class.getClassLoader().getResource("datasets/").getFile());

    @Test
    public void testClassificationOnUCRData() {
        // the relative path to the datasetsArray
        ClassLoader classLoader = SFAWordsTest.class.getClassLoader();
        for (DataSet dataSet : getDataSets()) {
            MDClassifier classifier = null;
            try {
                classifier = trainClassifier(dataSet);
            } catch (IOException e) {
                e.printStackTrace();
            }
            assertNotNull(classifier);
        }
    }


    public void testSave() throws IOException {
        DataSet dataSet = this.getDataSets().get(0);
        testSaveLoadGivesEqualTestResults(dataSet);
    }

    private void testSaveLoadGivesEqualTestResults(DataSet dataSet) throws IOException {
        /*MDClassifier classifier=trainClassifier(dataSet);
        File file=createTempClassifierFile();
        //classifier.save(file);
        Classifier loadedClassifier = Classifier.load(file);
        Assert.assertNotNull(loadedClassifier);
        checkEqualResultsOfClassifiers(dataSet, classifier, loadedClassifier);*/
    }

    private void checkEqualResultsOfClassifiers(DataSet dataSet, MDClassifier classifier, MDClassifier loadedClassifier) {
        /*MultiDimTimeSeries[] samples = MultiDimTimeSeriesLoader.loadDataset(getFirstTrainFile(dataSet));
        MDClassifier.Predictions loadedScore = loadedClassifier.score(samples);
        MDClassifier.Predictions score = classifier.score(samples);

        Assert.assertArrayEquals(loadedScore.labels, score.labels);
        Assert.assertEquals(loadedScore.correct.get(), score.correct.get());*/
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

    protected MDClassifier trainClassifier(DataSet dataSet) throws IOException {
        String[] sources = new String[]{"X", "Y", "Z"};
        String labals_string = new String("LABELS");


        // TRAIN
        ArrayList<File> files_train = new ArrayList<>();
        File file_labels_train = getFile(dataSet, "LABELS", TRAIN_FILE);
        for (String source_item : sources) {
            files_train.add(getFile(dataSet, source_item, TRAIN_FILE));
        }

        // TEST
        ArrayList<File> files_test = new ArrayList<>();
        File file_labels_test = getFile(dataSet, "LABELS", TEST_FILE);
        for (String source_item : sources) {
            files_test.add(getFile(dataSet, source_item, TEST_FILE));
        }

        MDClassifier classifier = null;
        MDClassifier.DEBUG = true;

        files_test.toArray(new File[files_test.size()]);

        // Load the train/test splits
        MultiDimTimeSeries[] testSamples = MultiDimTimeSeriesLoader.loadDataset(files_test.toArray(new File[]{}),file_labels_test);
        MultiDimTimeSeries[] trainSamples = MultiDimTimeSeriesLoader.loadDataset(files_train.toArray(new File[]{}),file_labels_train);

        classifier = initClassifier();
        MDClassifier.Score scoreW = classifier.eval(trainSamples, testSamples);
         /*   assertEquals("testing result of " +
                    dataSet.name+" does NOT match",
                    dataSet.testingAccuracy,
                    scoreW.getTestingAccuracy(),
                    DELTA);
            assertEquals("training result of "+dataSet.name+" does NOT match",
                    dataSet.trainingAccuracy,
                    scoreW.getTrainingAccuracy(),
                    DELTA);
            System.out.println(scoreW.toString());*/


        return classifier;
    }

    protected File getFile(DataSet dataSet, String source_string, String type_source) {
        File result = null;
        File dataSetDirectory = new File(DATASETS_DIRECTORY.getAbsolutePath() + "/" + dataSet.name);
        File[] filter = dataSetDirectory.listFiles(pathname -> pathname.getName().toUpperCase().startsWith(source_string));
        for (File file_selected : filter) {
            if (file_selected.getName().toUpperCase().endsWith(type_source)) {
                result = file_selected;
            }
        }
        return result;
    }

    private File[] getTrainFilesFromDir(File dataSetDirectory) {
        return dataSetDirectory.listFiles(pathname -> pathname.getName().toUpperCase().endsWith("TRAIN"));
    }

    protected abstract List<DataSet> getDataSets();


    protected abstract MDClassifier initClassifier();
}
