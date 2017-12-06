package har.classification;

import har.classification.AbstractClassifierMDTest;
import har.classification.MDClassifier;
import har.classification.WEASELMDClassifier;

import java.util.ArrayList;
import java.util.List;

public class WEASELMDClassifierTest extends AbstractClassifierMDTest {
    @Override
    protected List<DataSet> getDataSets() {
        List<DataSet> dataSets=new ArrayList<>();

        dataSets.add(new DataSet("UCI/UCI-HAR-TOTAL-REAL", 1.0, 1.0));

        return dataSets;
    }

    @Override
    protected MDClassifier initClassifier() {
        return new WEASELMDClassifier();
    }
}
