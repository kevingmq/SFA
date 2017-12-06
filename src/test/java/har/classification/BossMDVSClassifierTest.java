package har.classification;

import har.classification.AbstractClassifierMDTest;
import har.classification.BOSSMDVSClassifier;
import har.classification.MDClassifier;

import java.util.ArrayList;
import java.util.List;

public class BossMDVSClassifierTest extends AbstractClassifierMDTest {
    @Override
    protected List<DataSet> getDataSets() {
        List<DataSet> dataSets=new ArrayList<>();
        //Coffee;BOSS VS;1.0;1.0
        dataSets.add(new DataSet("UCI/UCI-HAR-TOTAL-REAL", 1.0, 1.0));

        return dataSets;
    }

    @Override
    protected MDClassifier initClassifier() {
        return new BOSSMDVSClassifier();
    }
}
