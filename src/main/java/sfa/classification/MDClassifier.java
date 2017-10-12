package sfa.classification;

import java.io.IOException;
import java.text.MessageFormat;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import com.carrotsearch.hppc.*;
import sfa.timeseries.MultiDimTimeSeries;
import com.carrotsearch.hppc.cursors.ObjectCursor;
import com.carrotsearch.hppc.cursors.FloatCursor;
import com.carrotsearch.hppc.cursors.IntCursor;
import sfa.timeseries.TimeSeries;

public abstract class MDClassifier {
    transient ExecutorService exec;

    public static boolean[] NORMALIZATION = new boolean[]{true, false};

    public static boolean DEBUG = true;
    public static boolean ENSEMBLE_WEIGHTS = true;

    public static int threads = 1;

    protected int[][] testIndices;
    protected int[][] trainIndices;
    public static int folds = 10;

    protected static int MAX_WINDOW_LENGTH = 250;

    // Blocks for parallel execution
    public final int BLOCKS = 8;

    static {
        Runtime runtime = Runtime.getRuntime();
        if (runtime.availableProcessors() <= 4) {
            threads = runtime.availableProcessors() - 1;
        } else {
            threads = runtime.availableProcessors();
        }
    }

    public MDClassifier() {
        this.exec = Executors.newFixedThreadPool(threads);
    }

    /**
     * Invokes {@code shutdown} when this executor is no longer
     * referenced and it has no threads.
     */
    protected void finalize() {
        if (exec != null) {
            exec.shutdown();
        }
    }

    /**
     * Build a classifier from the a training set with class labels.
     *
     * @param trainSamples The training set
     * @return The accuracy on the train-samples
     */
    public abstract Score fit(final MultiDimTimeSeries[] trainSamples);

    /**
     * The predicted classes and accuracies of an array of samples.
     *
     * @param testSamples The passed set
     * @return The predictions for each passed sample and the test accuracy.
     */
    public abstract Predictions score(final MultiDimTimeSeries[] testSamples);

    /**
     * The predicted classes of an array of samples.
     *
     * @param testSamples The passed set
     * @return The predictions for each passed sample.
     */
    public abstract Double[] predict(final MultiDimTimeSeries[] testSamples);

    /**
     * Performs training and testing on a set of train- and test-samples.
     *
     * @param trainSamples The training set
     * @param testSamples  The training set
     * @return The accuracy on the test- and train-samples
     */
    public abstract Score eval(
            final MultiDimTimeSeries[] trainSamples, final MultiDimTimeSeries[] testSamples);

    protected Predictions evalLabels(TimeSeries[] testSamples, Double[] labels) {
        int correct = 0;
        for (int ind = 0; ind < testSamples.length; ind++) {
            correct += compareLabels(labels[ind],(testSamples[ind].getLabel()))? 1 : 0;
        }
        return new Predictions(labels, correct);
    }

    public static class Words {
        public static int binlog(int bits) {
            int log = 0;
            if ((bits & 0xffff0000) != 0) {
                bits >>>= 16;
                log = 16;
            }
            if (bits >= 256) {
                bits >>>= 8;
                log += 8;
            }
            if (bits >= 16) {
                bits >>>= 4;
                log += 4;
            }
            if (bits >= 4) {
                bits >>>= 2;
                log += 2;
            }
            return log + (bits >>> 1);
        }

        public static long createWord(short[] words, int features, byte usedBits) {
            return fromByteArrayOne(words, features, usedBits);
        }

        /**
         * Returns a long containing the values in bytes.
         *
         * @param bytes
         * @param to
         * @param usedBits
         * @return
         */
        public static long fromByteArrayOne(short[] bytes, int to, byte usedBits) {
            int shortsPerLong = 60 / usedBits;
            to = Math.min(bytes.length, to);

            long bits = 0;
            int start = 0;
            long shiftOffset = 1;
            for (int i = start, end = Math.min(to, shortsPerLong + start); i < end; i++) {
                for (int j = 0, shift = 1; j < usedBits; j++, shift <<= 1) {
                    if ((bytes[i] & shift) != 0) {
                        bits |= shiftOffset;
                    }
                    shiftOffset <<= 1;
                }
            }

            return bits;
        }
    }

    public static class Model implements Comparable<Model> {

        public String name;
        public int windowLength;
        public boolean normed;

        public Score score;

        public Model(
                String name,
                int testing,
                int testSize,
                int training,
                int trainSize,
                boolean normed,
                int windowLength
        ) {
            this(name, new Score(name,testing,testSize,training,trainSize,windowLength),normed,windowLength);
        }

        public Model(
                String name,
                Score score,
                boolean normed,
                int windowLength
        ) {
            this.name = name;
            this.score = score;
            this.normed = normed;
            this.windowLength = windowLength;
        }

        @Override
        public String toString() {
            return score.toString();
        }

        public int compareTo(Model bestScore) { return this.score.compareTo(bestScore.score); }

    }


    public static class Score implements Comparable<Score> {
        public String name;
        public int training;
        public int trainSize;
        public int testing;
        public int testSize;
        public int windowLength;

        public Score() {
        }

        public Score(
                String name,
                int testing,
                int testSize,
                int training,
                int trainSize,
                int windowLength
        ) {
            this.name = name;
            this.training = training;
            this.trainSize = trainSize;
            this.testing = testing;
            this.testSize = testSize;
            this.windowLength = windowLength;
        }

        public double getTestingAccuracy() {
            return 1 - formatError(testing, testSize);
        }

        public double getTrainingAccuracy() {
            return 1 - formatError((int) training, trainSize);
        }

        @Override
        public String toString() {
            double test = getTestingAccuracy();
            double train = getTrainingAccuracy();

            return this.name + ";" + train + ";" + test;
        }


        public int compareTo(Score bestScore) {
            if (this.training > bestScore.training
                    || this.training == bestScore.training
                    && this.windowLength > bestScore.windowLength // on a tie, prefer the one with the larger window-length
                    ) {
                return 1;
            }
            return -1;
        }

        public void clear() {
            this.testing = 0;
            this.training = 0;
        }
    }

    public static class Predictions {

        public Double[] labels;
        public AtomicInteger correct;

        public Predictions(Double[] labels, int bestCorrect) {
            this.labels = labels;
            this.correct = new AtomicInteger(bestCorrect);
        }
    }

    public static void outputResult(int correct, long time, int testSize) {
        double error = formatError(correct, testSize);
        //String errorStr = MessageFormat.format("{0,number,#.##%}", error);
        String correctStr = MessageFormat.format("{0,number,#.##%}", 1 - error);

        System.out.print("Correct:\t");
        System.out.print("" + correctStr + "");
        System.out.println("\tTime: \t" + (System.currentTimeMillis() - time) / 1000.0 + " s");
    }

    public static double formatError(int correct, int testSize) {
        return Math.round(1000 * (testSize - correct) / (double) (testSize)) / 1000.0;
    }

    /*public static void outputConfusionMatrix(ObjectObjectOpenHashMap<String, ObjectLongOpenHashMap> matrix) {
        try {
            int rows = matrix.size();
            List<String> labels = new ArrayList(rows);

            for (ObjectCursor<String> actual_class : matrix.keys()) {
                labels.add(actual_class.value);

            }
            Collections.sort(labels, ALPHABETICAL_ORDER);

            int columns = rows;
            String str = "\t";
            String str2 = "\t";
            for (String l : labels){
                str += l + "\t";
                str2 += '-' + "\t";
            }
            System.out.println(str + "");
            System.out.println(str2 + "");
            str = "|\t";


            for (int i = 0; i < rows; i++) {
                for(int j = 0; j < columns;j++) {
                    str +=  matrix.get(labels.get(i)).get(labels.get(j)) + "\t";

                }
                str = labels.get(i) + str;
                System.out.println(str + "|");
                str = "|\t";
            }

        } catch (Exception e) {
            System.out.println("Matrix is empty!!");
        }
    }*/



//    public static Map<String, LinkedList<Integer>> splitByLabel(TimeSeries[] samples) {
//        Map<String, LinkedList<Integer>> elements = new HashMap<String, LinkedList<Integer>>();
//
//        for (int i = 0; i < samples.length; i++) {
//            String label = samples[i].getLabel();
//            if (!label.trim().isEmpty()) {
//                LinkedList<Integer> sameLabel = elements.get(label);
//                if (sameLabel == null) {
//                    sameLabel = new LinkedList<Integer>();
//                    elements.put(label, sameLabel);
//                }
//                sameLabel.add(i);
//            }
//        }
//        return elements;
//    }

    public static class Pair<E, T> {

        public E key;
        public T value;

        public Pair(E e, T t) {
            this.key = e;
            this.value = t;
        }

        public static <E, T> Pair<E, T> create(E e, T t) {
            return new Pair<E, T>(e, t);
        }

        @Override
        public int hashCode() {
            return this.key.hashCode();
        }

        @SuppressWarnings("unchecked")
        @Override
        public boolean equals(Object obj) {
            return this.key.equals(((Pair<E, T>) obj).key);
        }
    }


    protected boolean compareLabels(Double label1, Double label2) {
        // compare 1.0000 to 1.0 in String returns false, hence the conversion to double
        return label1 != null && label2 != null && label1.equals(label2);
    }

    protected <E extends Model> Ensemble<E> filterByFactor(
            List<E> results,
            int correctTraining,
            double factor) {

        // sort descending
        Collections.sort(results, Collections.reverseOrder());

        // only keep best scores
        List<E> model = new ArrayList<>();
        for (E score : results) {
            if (score.score.training >= correctTraining * factor) { // all with same score
                model.add(score);
            }
        }

        return new Ensemble<>(model);
    }

    protected Double[] score(
            final String name,
            final MultiDimTimeSeries[] samples,
            final List<Pair<Double, Integer>>[] labels,
            final List<Integer> currentWindowLengths) {

        Double[] predictedLabels = new Double[samples.length];
        //HashSet<String> uniqueLabels = uniqueClassLabels(samples); // OLHO somente as labels de TEST
        //ObjectObjectOpenHashMap<String, ObjectLongOpenHashMap> confusionMatrix = new ObjectObjectOpenHashMap<String, ObjectLongOpenHashMap>(uniqueLabels.size());
        //initConfusionMatrix(confusionMatrix, uniqueLabels);

        int correctTesting = 0;
        for (int i = 0; i < labels.length; i++) {
            Map<Double, Long> counts = new HashMap<>();

            for (Pair<Double, Integer> k : labels[i]) {
                if (k != null && k.key != null) {
                    Double label = k.key;
                    Long count = counts.get(label);
                    long increment = ENSEMBLE_WEIGHTS ? k.value : 1;
                    count = (count == null) ? increment : count + increment;
                    counts.put(label, count);

                }
            }
            /*
            if (samples[i].getLabel().equals(maxLabel)) {
                correctTesting++;
                confusionMatrix.get(maxLabel).putOrAdd(maxLabel, (long) 1, (long) 1);
            } else {
                confusionMatrix.get(samples[i].getLabel()).putOrAdd(maxLabel, (long) 1, (long) 1);
            }*/
            long maxCount = -1;
            for (Entry<Double, Long> e : counts.entrySet()) {
                if (predictedLabels[i] == null
                        || maxCount < e.getValue()
                        || maxCount == e.getValue()  // break ties
                        && Double.valueOf(predictedLabels[i]) <= Double.valueOf(e.getKey())
                        ) {
                    maxCount = e.getValue();
                    // maxCounts[i] = maxCount;
                    predictedLabels[i] = e.getKey();
                }
            }
        }

        if (DEBUG) {
            System.out.println(name + " Testing with " + currentWindowLengths.size() + " models:\t");
            System.out.println(currentWindowLengths.toString() + "\n");
            //outputResult(correctTesting, startTime, samples.length);
           // outputConfusionMatrix(confusionMatrix);
        }
        return predictedLabels;
    }

    protected Integer[] getWindowsBetween(int minWindowLength, int maxWindowLength) {
        List<Integer> windows = new ArrayList<>();
        for (int windowLength = maxWindowLength; windowLength >= minWindowLength; windowLength--) {
            windows.add(windowLength);
        }
        return windows.toArray(new Integer[]{});
    }

    protected int getMax(MultiDimTimeSeries[] samples, int MAX_WINDOW_SIZE) {
        int max = MAX_WINDOW_SIZE;
        for (MultiDimTimeSeries ts : samples) {
            max = Math.min(ts.getLength(), max);
        }
        return max;
    }

    protected static Set<Double> uniqueClassLabels(MultiDimTimeSeries[] ts) {
        Set<Double> labels = new HashSet<>();
        for (MultiDimTimeSeries t : ts) {
            labels.add(t.getLabel());
        }
        return labels;
    }

    protected static double magnitude(FloatContainer values) {
        double mag = 0.0D;
        for (FloatCursor value : values) {
            mag = mag + value.value * value.value;
        }
        return Math.sqrt(mag);
    }

    protected static int[] createIndices(int length) {
        int[] indices = new int[length];
        for (int i = 0; i < length; i++) {
            indices[i] = i;
        }
        return indices;
    }

    protected void generateIndices(MultiDimTimeSeries[] samples) {
        IntArrayList[] sets = getStratifiedTrainTestSplitIndices(samples, folds);
        this.testIndices = new int[folds][];
        this.trainIndices = new int[folds][];
        for (int s = 0; s < folds; s++) {
            this.testIndices[s] = convertToInt(sets[s]);
            this.trainIndices[s] = convertToInt(sets, s);
        }
    }

    protected IntArrayList[] getStratifiedTrainTestSplitIndices(
            MultiDimTimeSeries[] samples,
            int splits) {

        HashMap<Double, IntArrayDeque> elements = new HashMap<>();

        for (int i = 0; i < samples.length; i++) {
            Double label = samples[i].getLabel();
            IntArrayDeque sameLabel = elements.get(label);
            if (sameLabel == null) {
                sameLabel = new IntArrayDeque();
                elements.put(label, sameLabel);
            }
            sameLabel.addLast(i);
        }

        // pick samples
        IntArrayList[] sets = new IntArrayList[splits];
        for (int i = 0; i < splits; i++) {
            sets[i] = new IntArrayList();
        }

        // all but one
        for (Entry<Double, IntArrayDeque> data : elements.entrySet()) {
            IntArrayDeque d = data.getValue();
            separate:
            while (true) {
                for (int s = 0; s < splits; s++) {
                    if (!d.isEmpty()) {
                        int dd = d.removeFirst();
                        sets[s].add(dd);
                    } else {
                        break separate;
                    }
                }
            }
        }

        return sets;
    }

    protected static int[] convertToInt(IntArrayList trainSet) {
        int[] train = new int[trainSet.size()];
        int a = 0;
        for (IntCursor i : trainSet) {
            train[a++] = i.value;
        }
        return train;
    }

    protected static int[] convertToInt(IntArrayList[] setToSplit, int exclude) {
        int count = 0;

        for (int i = 0; i < setToSplit.length; i++) {
            if (i != exclude) {
                count += setToSplit[i].size();
            }
        }

        int[] setData = new int[count];
        int a = 0;
        for (int i = 0; i < setToSplit.length; i++) {
            if (i != exclude) {
                for (IntCursor d : setToSplit[i]) {
                    setData[a++] = d.value;
                }
            }
        }

        return setData;
    }

   /* protected void initConfusionMatrix(
            final ObjectObjectOpenHashMap<String, ObjectLongOpenHashMap> matrix,
            final HashSet<String> uniqueLabels) {
        for (String label : uniqueLabels) {
            ObjectLongOpenHashMap stat = matrix.get(label);
            if (stat == null) {
                matrix.put(label, new ObjectLongOpenHashMap(uniqueLabels.size()));
            } else {
                if (stat != null) {
                    stat.clear();
                }
            }
        }
    }*/
}
