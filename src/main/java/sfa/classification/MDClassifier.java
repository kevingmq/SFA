package sfa.classification;
import java.io.File;
import java.io.IOException;
import java.text.MessageFormat;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicInteger;

import com.carrotsearch.hppc.*;
import sfa.timeseries.MultiDimTimeSeries;
import sfa.timeseries.MultiDimTimeSeriesLoader;
import com.carrotsearch.hppc.cursors.ObjectCursor;
import com.carrotsearch.hppc.cursors.FloatCursor;
import com.carrotsearch.hppc.cursors.IntCursor;
import sfa.timeseries.TimeSeries;

public class MDClassifier {
    public MultiDimTimeSeries[] testSamples;
    public MultiDimTimeSeries[] trainSamples;
    public static int threads = 4;
    public static boolean DEBUG = true;

    public static boolean[] NORMALIZATION = new boolean[]{true, false};


    public static boolean ENSEMBLE_WEIGHTS = true;


    public AtomicInteger correctTraining = new AtomicInteger(0);

    protected int[][] testIndices;
    protected int[][] trainIndices;
    public static int folds = 10;

    protected static int MAX_WINDOW_LENGTH = 250;

    // Blocks for parallel execution
    public final int BLOCKS = 8;

    static {
//        Runtime runtime = Runtime.getRuntime();
//        if (runtime.availableProcessors() <= 4) {
//            threads = runtime.availableProcessors() - 1;
//        } else {
//            threads = runtime.availableProcessors();
//        }
    }

    public MDClassifier(MultiDimTimeSeries[] train, MultiDimTimeSeries[] test) throws IOException {
        this.trainSamples = train;
        this.testSamples = test;
    }

    public abstract Score eval() throws IOException;

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

    public static class Score implements Comparable<Score> {

        public String name;
        public double training;
        public double testing;
        public boolean normed;
        public int windowLength;

        public Score(
                String name,
                double testing,
                double training,
                boolean normed,
                int windowLength
        ) {
            this.name = name;
            this.training = training;
            this.testing = testing;
            this.normed = normed;
            this.windowLength = windowLength;
        }

        @Override
        public String toString() {
            return this.name + ";" + this.training + ";" + this.testing;
        }

        public int compareTo(Score bestScore) {
            if (this.training > bestScore.training
                    || this.training == bestScore.training
                    && this.windowLength > bestScore.windowLength) {
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

        public String[] labels;
        public AtomicInteger correct;

        public Predictions(String[] labels, int bestCorrect) {
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

    public static void outputConfusionMatrix(ObjectObjectOpenHashMap<String, ObjectLongOpenHashMap> matrix) {
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
    }

    private static Comparator<String> ALPHABETICAL_ORDER = new Comparator<String>() {
        public int compare(String str1, String str2) {
            int res = String.CASE_INSENSITIVE_ORDER.compare(str1, str2);
            if (res == 0) {
                res = str1.compareTo(str2);
            }
            return res;
        }
    };

    public static double formatError(int correct, int testSize) {
        double error = Math.round(1000 * (testSize - correct) / (double) (testSize)) / 1000.0;
        return error;
    }

    @SuppressWarnings("unchecked")
    public static TimeSeries[][] getStratifiedSplits(
            TimeSeries[] samples,
            int splits) {

        Map<String, LinkedList<Integer>> elements = splitByLabel(samples);

        // pick samples
        double trainTestSplit = 1.0 / (double) splits;
        ArrayList<TimeSeries>[] sets = new ArrayList[splits];
        for (int s = 0; s < splits; s++) {
            sets[s] = new ArrayList<TimeSeries>();
            for (Entry<String, LinkedList<Integer>> data : elements.entrySet()) {
                int count = (int) (data.getValue().size() * trainTestSplit);
                int i = 0;
                while (!data.getValue().isEmpty()
                        && i <= count) {
                    sets[s].add(samples[data.getValue().remove()]);
                    i++;
                }
            }
        }

        ArrayList<TimeSeries> testSet = new ArrayList<TimeSeries>();
        for (List<Integer> indices : elements.values()) {
            for (int index : indices) {
                testSet.add(samples[index]);
            }
        }

        TimeSeries[][] data = new TimeSeries[splits][];
        for (int s = 0; s < splits; s++) {
            data[s] = sets[s].toArray(new TimeSeries[]{});
        }

        return data;
    }

    public static Map<String, LinkedList<Integer>> splitByLabel(TimeSeries[] samples) {
        Map<String, LinkedList<Integer>> elements = new HashMap<String, LinkedList<Integer>>();

        for (int i = 0; i < samples.length; i++) {
            String label = samples[i].getLabel();
            if (!label.trim().isEmpty()) {
                LinkedList<Integer> sameLabel = elements.get(label);
                if (sameLabel == null) {
                    sameLabel = new LinkedList<Integer>();
                    elements.put(label, sameLabel);
                }
                sameLabel.add(i);
            }
        }
        return elements;
    }

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

    public int score(
            final String name,
            final MultiDimTimeSeries[] samples,
            long startTime,
            final List<Pair<String, Double>>[] labels,
            final List<Integer> currentWindowLengths) {
        HashSet<String> uniqueLabels = uniqueClassLabels(samples); // OLHO somente as labels de TEST
        ObjectObjectOpenHashMap<String, ObjectLongOpenHashMap> confusionMatrix = new ObjectObjectOpenHashMap<String, ObjectLongOpenHashMap>(uniqueLabels.size());
        initConfusionMatrix(confusionMatrix, uniqueLabels);
        int correctTesting = 0;
        for (int i = 0; i < labels.length; i++) {

            String maxLabel = "";
            double maxCount = 0.0;

            HashMap<String, Double> counts = new HashMap<String, Double>();

            for (Pair<String, Double> k : labels[i]) {
                if (k != null && k.key != null) {
                    String s = k.key;
                    Double count = counts.get(s);
                    double increment = ENSEMBLE_WEIGHTS ? k.value : 1;
                    count = (count == null) ? increment : count + increment;
                    counts.put(s, count);
                    if (maxCount < count
                            || maxCount == count && maxLabel.compareTo(s) < 0) {
                        maxCount = count;
                        maxLabel = s;
                    }
                }
            }
            if (samples[i].getLabel().equals(maxLabel)) {
                correctTesting++;
                confusionMatrix.get(maxLabel).putOrAdd(maxLabel, (long) 1, (long) 1);
            } else {
                confusionMatrix.get(samples[i].getLabel()).putOrAdd(maxLabel, (long) 1, (long) 1);
            }

        }

        if (DEBUG) {
            System.out.println(name + " Testing with " + currentWindowLengths.size() + " models:\t");
            outputResult(correctTesting, startTime, samples.length);
            outputConfusionMatrix(confusionMatrix);
        }
        return correctTesting;
    }

    public int getMax(MultiDimTimeSeries[] samples, int MAX_WINDOW_SIZE) {
        int max = MAX_WINDOW_SIZE;
        for (MultiDimTimeSeries ts : samples) {
            max = Math.min(ts.getLength(), max);
        }
        return max;
    }

    protected static HashSet<String> uniqueClassLabels(MultiDimTimeSeries[] ts) {
        HashSet<String> labels = new HashSet<String>();
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

    protected void generateIndices() {
        IntArrayList[] sets = getStratifiedTrainTestSplitIndices(this.trainSamples, folds);
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

        HashMap<String, IntArrayDeque> elements = new HashMap<String, IntArrayDeque>();

        for (int i = 0; i < samples.length; i++) {
            String label = samples[i].getLabel();
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
        for (Entry<String, IntArrayDeque> data : elements.entrySet()) {
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

    protected void initConfusionMatrix(
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
    }
}
