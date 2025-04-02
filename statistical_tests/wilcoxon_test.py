import numpy as np
from scipy import stats

ALPHA = 0.05

def cohens_d_paired(x, y):
    """Cohen's d for paired samples"""
    diff = np.array(y) - np.array(x)
    return np.mean(diff) / np.std(diff, ddof=1)

def hypothesis_1(hybrid_metrics: dict, baseline_metrics: dict):
    """
    Hypothesis 1:
    H1: The proposed hybrid approach will demonstrate
    significantly higher person Re-ID performance
    compared to a conventional CNN-based system.

    H0: he proposed hybrid approach does not
    demonstrate a statistically significant improvement in
    person Re-ID performance compared to a
    conventional CNN-based system.

    Parameters:
        hybrid_metrics (dict): Contains arrays for 'rank1', 'rank5', and 'ap' for
        the hybrid approach.
        baseline_metrics (dict): Contains arrays for the same keys for the baseline approach.
    """
    print("Hypothesis 1: Hybrid vs. Baseline")
    for metric in ['rank1', 'rank5', 'ap']:
        hybrid_arr = hybrid_metrics[metric]
        baseline_arr = baseline_metrics[metric]

        # Compute mean and standard deviation
        hybrid_mean = np.mean(hybrid_arr)
        hybrid_std = np.std(hybrid_arr)
        baseline_mean = np.mean(baseline_arr)
        baseline_std = np.std(baseline_arr)

        cohens_d = cohens_d_paired(baseline_arr, hybrid_arr)

        print(f"{metric.upper()} - Hybrid cropped: "
              f"mean = {hybrid_mean:.3f}, "
              f"std = {hybrid_std:.3f}")
        print(f"{metric.upper()} - Resnet50 cropped: "
              f"mean = {baseline_mean:.3f}, "
              f"std = {baseline_std:.3f}")

        try:
            result = stats.wilcoxon(hybrid_arr, baseline_arr)
            stat = result.statistic
            p_val = result.pvalue
            print(f"{metric.upper()}: Wilcoxon statistic = {stat:.3f}, "
                  f"p-value = {p_val:.4f}, "
                  f"d = {cohens_d:.4f}")
            if p_val < ALPHA:
                print(f"  -> Reject H0 for {metric.upper()}: "
                      f"significant difference detected.\n")
            else:
                print(f"  -> Fail to reject H0 for {metric.upper()}: "
                      f"no significant difference.\n")
        except Exception as e:
            print(f"Error testing {metric.upper()}: {e}")


def hypothesis_2(hybrid_uncropped_metrics: dict,
                 baseline_uncropped_metrics: dict,
                 hybrid_cropped_metrics: dict):
    """
    Hypothesis 2:
    H1: The hybrid approach will outperform a
    traditional CNN-based approach in terms of person
    Re-ID accuracy when processing un-cropped images
    due to its ability to isolate meaningful object
    segments.

    H0: here is no statistically significant difference in
    person Re-ID performance between the hybrid
    method and the CNN-based method when using
    un-cropped images.

    Parameters:
        hybrid_uncropped_metrics, hybrid_cropped_metrics, cnn_cropped_metrics (dict):
            Each dictionary contains per-query arrays for 'rank1', 'rank5', and 'ap'.
    """
    print("Hypothesis 2: Hybrid vs. Baseline (Un-cropped)")
    for metric in ['rank1', 'rank5', 'ap']:
        hybrid_uncropped_arr = hybrid_uncropped_metrics[metric]
        baseline_uncropped_arr = baseline_uncropped_metrics[metric]

        # Compute mean and standard deviation
        hybrid_uncropped_mean = np.mean(hybrid_uncropped_arr)
        hybrid_uncropped_std = np.std(hybrid_uncropped_arr)
        baseline_uncropped_mean = np.mean(baseline_uncropped_arr)
        baseline_uncropped_std = np.std(baseline_uncropped_arr)

        cohens_d = cohens_d_paired(baseline_uncropped_arr, hybrid_uncropped_arr)

        print(f"{metric.upper()} - Hybrid un-cropped: "
              f"mean = {hybrid_uncropped_mean:.3f}, "
              f"std = {hybrid_uncropped_std:.3f}")
        print(f"{metric.upper()} - Resnet50 un-cropped: "
              f"mean = {baseline_uncropped_mean:.3f}, "
              f"std = {baseline_uncropped_std:.3f}")

        try:
            result = stats.wilcoxon(hybrid_uncropped_arr, baseline_uncropped_arr)
            stat = result.statistic
            p_val = result.pvalue
            print(f"{metric.upper()} (Cropped): "
                  f"Wilcoxon statistic = {stat:.3f}, "
                  f"p-value = {p_val:.4f}, "
                  f"d = {cohens_d:.4f}")
            if p_val < ALPHA:
                print(f"  -> Reject H0 for {metric.upper()} (Cropped): "
                      f"significant difference detected.\n")
            else:
                print(f"  -> Fail to reject H0 for {metric.upper()} (Cropped): "
                      f"no significant difference.\n")
        except Exception as e:
            print(f"Error testing {metric.upper()} (Cropped): {e}")

    print("Hypothesis 2 Extra Hybrid Analysis: Hybrid (Cropped) vs. Hybrid (Uncropped)")
    for metric in ['rank1', 'rank5', 'ap']:
        try:
            hybrid_uncropped_arr = hybrid_uncropped_metrics[metric]
            hybrid_cropped_arr = hybrid_cropped_metrics[metric]

            # Compute mean and standard deviation
            hybrid_uncropped_mean = np.mean(hybrid_uncropped_arr)
            hybrid_uncropped_std = np.std(hybrid_uncropped_arr)
            hybrid_cropped_mean = np.mean(hybrid_cropped_arr)
            hybrid_cropped_std = np.std(hybrid_cropped_arr)

            cohens_d = cohens_d_paired(hybrid_uncropped_arr, hybrid_cropped_arr)

            print(f"{metric.upper()} - Hybrid un-cropped: "
                  f"mean = {hybrid_uncropped_mean:.3f}, "
                  f"std = {hybrid_uncropped_std:.3f}")
            print(f"{metric.upper()} - Hybrid cropped: "
                  f"mean = {hybrid_cropped_mean:.3f}, "
                  f"std = {hybrid_cropped_std:.3f}")

            result = stats.wilcoxon(hybrid_uncropped_arr, hybrid_cropped_arr)
            stat = result.statistic
            p_val = result.pvalue

            print(f"{metric.upper()} (Uncropped): "
                  f"Wilcoxon statistic = {stat:.3f}, "
                  f"p-value = {p_val:.4f}, "
                  f"d = {cohens_d:.4f}")
            if p_val < ALPHA:
                print(f"  -> Reject H0 for {metric.upper()} (Uncropped): "
                      f"significant difference detected.\n")
            else:
                print(f"  -> Fail to reject H0 for {metric.upper()} (Uncropped): "
                      f"no significant difference.\n")
        except Exception as e:
            print(f"Error testing {metric.upper()} (Uncropped): {e}")



def hypothesis_3(clahe_metrics_uncropped: dict,
                 non_clahe_metrics_uncropped: dict,
                 clahe_metrics_cropped: dict,
                 non_clahe_metrics_cropped: dict):
    """
    Hypothesis 3:
    H1: Incorporating CLAHE into the Re-ID pipeline
    will significantly improve person matching accuracy
    under varying illumination.

    H0: CLAHE does not produce a statistically
    significant improvement in person Re-ID
    performance under varying illumination conditions.

    Parameters:
        clahe_metrics (dict): Per-query arrays for 'rank1', 'rank5', and 'ap' with CLAHE.
        non_clahe_metrics (dict): Per-query arrays for the same metrics without CLAHE.
    """
    print("Hypothesis 3 cropped: CLAHE vs. No CLAHE on cropped images")
    for metric in ['rank1', 'rank5', 'ap']:
        try:
            clahe_cropped_arr = clahe_metrics_cropped[metric]
            non_clahe_cropped_arr = non_clahe_metrics_cropped[metric]

            # Compute mean and standard deviation
            clahe_cropped_mean = np.mean(clahe_cropped_arr)
            clahe_cropped_std = np.std(clahe_cropped_arr)
            non_clahe_cropped_mean = np.mean(non_clahe_cropped_arr)
            non_clahe_cropped_std = np.std(non_clahe_cropped_arr)

            cohens_d = cohens_d_paired(non_clahe_cropped_arr, clahe_cropped_arr)

            print(f"{metric.upper()} - CLAHE cropped: "
                  f"mean = {clahe_cropped_mean:.3f}, "
                  f"std = {clahe_cropped_std:.3f}")
            print(f"{metric.upper()} - Non CLAHE cropped: "
                  f"mean = {non_clahe_cropped_mean:.3f}, "
                  f"std = {non_clahe_cropped_std:.3f}")

            result = stats.wilcoxon(clahe_cropped_arr, non_clahe_cropped_arr)
            stat = result.statistic
            p_val = result.pvalue
            print(f"{metric.upper()}: Wilcoxon statistic = {stat:.3f}, "
                  f"p-value = {p_val:.4f}, "
                  f"d = {cohens_d:.4f}")
            if p_val < ALPHA:
                print(f"  -> Reject H0 for {metric.upper()}: "
                      f"significant difference detected\n")
            else:
                print(f"  -> Fail to reject H0 for {metric.upper()}: "
                      f"no significant difference.\n")
        except Exception as e:
            print(f"Error testing {metric.upper()}: {e}")

    print("Hypothesis 3 un-cropped: CLAHE vs. No CLAHE on un-cropped images")
    for metric in ['rank1', 'rank5', 'ap']:
        try:
            clahe_uncropped_arr = clahe_metrics_uncropped[metric]
            non_clahe_uncropped_arr = non_clahe_metrics_uncropped[metric]

            # Compute mean and standard deviation
            clahe_uncropped_mean = np.mean(clahe_uncropped_arr)
            clahe_uncropped_std = np.std(clahe_uncropped_arr)
            non_clahe_uncropped_mean = np.mean(non_clahe_uncropped_arr)
            non_clahe_uncropped_std = np.std(non_clahe_uncropped_arr)

            cohens_d = cohens_d_paired(non_clahe_uncropped_arr, clahe_uncropped_arr)

            print(f"{metric.upper()} - CLAHE un-cropped: "
                  f"mean = {clahe_uncropped_mean:.3f}, "
                  f"std = {clahe_uncropped_std:.3f}")
            print(f"{metric.upper()} - Non CLAHE un-cropped: "
                  f"mean = {non_clahe_uncropped_mean:.3f}, "
                  f"std = {non_clahe_uncropped_std:.3f}")

            result = stats.wilcoxon(clahe_uncropped_arr, non_clahe_uncropped_arr)
            stat = result.statistic
            p_val = result.pvalue
            print(f"{metric.upper()}: Wilcoxon statistic = {stat:.3f}, "
                  f"p-value = {p_val:.4f}, "
                  f"d = {cohens_d:.4f}")
            if p_val < ALPHA:
                print(f"  -> Reject H0 for {metric.upper()}: "
                      f"significant difference detected.\n")
            else:
                print(f"  -> Fail to reject H0 for {metric.upper()}: "
                      f"no significant difference.\n")
        except Exception as e:
            print(f"Error testing {metric.upper()}: {e}")


if __name__ == "__main__":
    # These are computed using evaluate_rank_map_per_query.
    hybrid_cropped_metrics = {
        'rank1': np.array([1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        'rank5': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        'ap': np.array([0.7962515262515262, 0.2559774249373881, 0.4977287702052811, 0.5070954138279141, 0.745610188150146, 1.0, 0.6670070251518214, 0.38224720922222655, 0.6616944407534255, 0.1735038272032443, 0.6994698911624377, 0.34409927754685093, 0.1273116995982374, 0.7086925918692836, 0.7086419578810883, 0.7563284929356356, 0.42689390211188855, 0.8154461279461279, 0.13461126357039044, 0.7570809958138573, 0.4475413362837298, 0.7220186606310286, 0.21977237581168474, 0.7508584587531956, 0.9248015873015872, 0.5207600397160836, 0.45516139009593265, 0.966184692500482, 1.0, 0.801427179058758, 0.7532238143873364, 0.0691189913498355, 0.6274271965070474, 0.7493663743663743, 0.13313857143755528, 0.40490471363396874, 0.6029156816390858, 0.6859265561905313, 0.19931485271133867, 0.6696760759730684, 0.24949728971948884, 0.9271811713191024, 0.6123199235527853, 0.2484088541105326, 0.9815204678362572, 0.6026714026714027, 0.5891336202797247, 0.4675357755963366, 0.7596149648391802, 0.9197396584896586, 0.7591652931985958, 0.3785645613207852, 0.3594188390902158, 0.68805602517141, 0.69730815764807, 0.9669202592279517, 0.9218623481781376, 0.8822956672956673, 0.5807540704909125, 0.9754901960784313, 0.17225915932032815, 0.8096537993596816, 0.7633889295935603, 0.7981081788818363, 0.28929575861196477, 0.2650711447712088, 0.8138407888407888, 0.09398674210326542, 0.27909650627737553, 0.5743410563045389, 0.3775584577568601, 0.5486441813547077, 0.8206399156399157, 0.3478777492199407, 0.5870993347540422, 0.9921568627450981, 0.9282114376851218, 0.8074774088263696, 0.7734789789127547, 0.7110536806188981, 0.5773318532813254, 0.8839667718156091, 0.8434143634143634, 0.6359578135893925, 0.33000091988826363, 0.42744407998360756, 0.7644817219817222, 0.9096516068497492, 0.7898949758177719, 0.48396355688875414, 0.50013472040984, 0.7118884862962127, 0.6831870399612333, 0.8181322872398389, 0.64649151628099, 0.4018024466127281, 0.8261377311377313, 0.7603200238238601, 0.9057279952016795, 0.44944427200009585])
    }
    baseline_cropped_metrics = {
        'rank1': np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]),
        'rank5': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        'ap': np.array([0.9585881509794554, 0.26883186120620634, 0.116813697227006, 0.6095255865108806, 0.283792530151119, 0.5994405594405594, 0.5896608946608947, 0.8423738770537817, 0.6626814476814477, 0.7554551494257375, 0.3805454863728133, 0.44012279659537556, 0.4497485784006398, 0.726201630118409, 0.749128490557062, 0.1696371265710069, 0.49123430486889597, 0.18281891215639642, 0.48300426657537704, 0.5531023302515606, 0.0380658466651037, 0.8135899494394027, 0.23796669086967243, 0.766265240136208, 0.9705128205128206, 0.3758857295168833, 0.3715583750576517, 0.9438775510204083, 0.8933470954657396, 0.46616006673245824, 0.6426622195878778, 0.5114059815361581, 0.11468150451863805, 0.3494180984240426, 0.6150775613275613, 0.37465651887492507, 0.06477403343441815, 0.31895093771180827, 0.6338615438259215, 0.790028528733988, 0.9598765432098766, 0.29588839656081056, 0.39707957187114407, 0.7376848988259687, 0.6670715441660389, 0.5426916236368594, 0.7846941755562443, 0.44533016906443634, 0.3413098023384436, 0.30321478587786965, 0.39615516814058793, 0.21204201181675703, 0.3756984417323936, 0.7347753261442651, 0.3804420344921416, 0.7665282639553197, 0.7025856496444731, 0.39888773085478296, 0.3972184409970826, 0.10474048902768508, 0.3766252284683863, 0.7154390516890516, 0.2613502850140322, 0.16936721738897553, 0.33091718470081694, 0.7650109123017713, 0.6769951454108252, 0.2107436830912268, 0.2707407359225472, 0.43496802435023857, 0.40537262138722513, 0.407142075293698, 0.47068413767627876, 0.605202558703041, 0.292912512325511, 0.8229918229918229, 0.5910647986841954, 0.7345132624267963, 0.11287811205058203, 0.8762866762866762, 0.5631969532078803, 0.6600422646880448, 0.6334760593412551, 0.7942989066217359, 0.4455887116617922, 0.4346071276753866, 0.6971298961790686, 0.5460068728645948, 0.3472563202819724, 0.211005118025467, 0.6366150860888509, 0.1891573302734497, 0.37839338990122223, 0.658678369597217, 0.3807569378983207, 0.3508435549649688, 0.6489752491914338, 0.24258756577600682, 0.6723298797136008, 0.14893915119729725])
    }
    hypothesis_1(hybrid_cropped_metrics, baseline_cropped_metrics)

    hybrid_uncropped_metrics = {
        'rank1': np.array([1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]),
        'rank5': np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        'ap': np.array([0.9509085213032581, 0.247104733780833, 0.14703302597652396, 0.37874517483197406, 0.42455314348448864, 0.9136577708006278, 0.6568515009894321, 0.3344968093647477, 0.8524207018791305, 0.15085162033852123, 0.48335034013605443, 0.14279786474896394, 0.16723457600720742, 0.6446045244297525, 0.5885221377510994, 0.6437891545644944, 0.3950101455481656, 0.6056345828084958, 0.45266492061811053, 0.9182357441811223, 0.06166223814499675, 0.7147135656026428, 0.24241013591813365, 0.7911136663997076, 0.7484922484922486, 0.488540912466351, 0.5077558801298296, 0.9366110080395795, 0.8986907768157767, 0.6917563034759099, 0.5914637239006988, 0.08747308035802862, 0.5534082921176288, 0.9592975651799182, 0.08222376220638351, 0.3530365373324491, 0.5341858626919603, 0.5661634496455925, 0.07152673410861722, 0.8266909062475564, 0.282890966365297, 0.6944911381637044, 0.5291627919723895, 0.23271153860003768, 0.97906162464986, 0.7025262972295643, 0.6012830845071001, 0.4023595118311937, 0.8791137052994142, 0.7448825366973633, 0.8471101507117046, 0.3422373253554147, 0.2001395474789532, 0.8190125152625153, 0.7609901536372123, 0.8293231341308266, 0.966923076923077, 0.8818495514147686, 0.4894548763326896, 0.6850559163059162, 0.1268822628600358, 0.8877666315166315, 0.8237412696236227, 0.797302473355105, 0.12913878852984223, 0.4749460924151188, 0.9092953342953343, 0.07282477247464583, 0.45102876775369183, 0.5068722622573818, 0.3839546500709504, 0.6335747155659905, 0.9108970658970659, 0.420436677899943, 0.6140646116450651, 0.8299552299552301, 0.6343945704220245, 0.6808495209284803, 0.772507580613133, 0.8581581796206913, 0.6498677663799614, 0.7004660154660155, 0.7965311377076084, 0.5431404769082918, 0.4353090265315307, 0.31659213453685464, 0.6528821338225362, 0.8384174171460698, 0.8060598077721367, 0.3828836793560057, 0.34237884618319403, 0.6402805723339814, 0.5892250286268063, 0.658382705233636, 0.5255168427261153, 0.08464593573980846, 0.6054252567551799, 0.8801065291777367, 0.7518243797588838, 0.3544801573170122])
    }

    baseline_uncropped_metrics = {
        'rank1': np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0]),
        'rank5': np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]),
        'ap': np.array([0.4572086838890615, 0.09993180020258957, 0.021005725458562493, 0.12677699174618492, 0.04006044075216541, 0.015699830619280465, 0.03656145136784719, 0.03879067581197648, 0.0847629322601469, 0.31654020637040525, 0.02951301841812795, 0.15375473955890712, 0.09535419445747875, 0.27021457958490486, 0.14179333702558264, 0.032524714329919935, 0.12169599089605235, 0.13527040177399774, 0.22548620126965976, 0.17087238004837416, 0.0170025363871212, 0.1960732843956614, 0.02314623016433871, 0.08402654360931736, 0.11778854226605108, 0.1543316940746238, 0.017472479490954002, 0.4135722337104471, 0.3010364835488886, 0.10759905640906929, 0.24180462257072796, 0.01660235077763641, 0.11564925710759243, 0.1490947801275707, 0.054733394931400746, 0.03470769197645878, 0.06745228606587482, 0.019486684342238554, 0.06139103403928787, 0.3378619364569663, 0.12905575609983172, 0.10673362494367905, 0.014865519445467306, 0.5206338787694083, 0.1384923535801278, 0.1595834200291425, 0.6719817507301286, 0.16573076925717148, 0.18558903644080382, 0.13832162799628064, 0.18780001303635438, 0.3591959152604086, 0.06213321940113948, 0.6830681865392875, 0.25834144405413073, 0.547306832426922, 0.2528898264106304, 0.5169858513889031, 0.24604945673766368, 0.4102863693270413, 0.20616638259635156, 0.3288880299943787, 0.07573692206622436, 0.03844105827714772, 0.2859441290742205, 0.1540151427679682, 0.40828291994353694, 0.01928029850286598, 0.10097601200587011, 0.17526253488856722, 0.05884470774732857, 0.1348075418508085, 0.18927386091733284, 0.03476006809354434, 0.05344873710204289, 0.05161826541045997, 0.28073244058432345, 0.13947900020943918, 0.04946636636856606, 0.3031791059970968, 0.10466524990790899, 0.23172827278014274, 0.2536321000310539, 0.04370553963629573, 0.1871546332398943, 0.15672812906049943, 0.6260757095350772, 0.08794025742808718, 0.09302699405179071, 0.24493480543824397, 0.10701572471647366, 0.10970370877903042, 0.4570898874762767, 0.052888504390101135, 0.2848880709519144, 0.058128282596637305, 0.46117579559255667, 0.08450982436515293, 0.3472320281523682, 0.04418748483143996])
    }

    hypothesis_2(hybrid_uncropped_metrics, baseline_uncropped_metrics, hybrid_cropped_metrics)

    hybrid_uncropped_clahe = hybrid_uncropped_metrics

    hybrid_uncropped_no_clahe = {
        'rank1': np.array([1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        'rank5': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        'ap': np.array([0.8970299635005516, 0.2552319442496927, 0.2561996307391061, 0.6841367699988389, 0.7341152437306283, 0.9821428571428571, 0.7346635202404433, 0.4858503576851127, 0.9833333333333333, 0.3401150793650793, 0.828324110124439, 0.26448606788308887, 0.3444115369607345, 0.5947868814341286, 0.8823382195146903, 0.7522417726752091, 0.28449103309552987, 0.791991341991342, 0.3009108492117332, 0.882371398442827, 0.6726044226044227, 0.625957356688957, 0.306582343040421, 0.8052934365100142, 0.9707362082362082, 0.39184923187855153, 0.47058139744112265, 0.7304225537097109, 0.9880952380952381, 0.867178268354739, 0.6979566772193501, 0.07179148419305488, 0.5753140631581438, 0.591232859066097, 0.19884853886655657, 0.7138077124340861, 0.5643321431136558, 0.8214002392881702, 0.20060663928767916, 0.9226234292023766, 0.20917801794994775, 0.6030142123054494, 0.5700068022547015, 0.4307246404328785, 0.9866269841269841, 0.637870832156963, 0.8392419465273692, 0.20690518330710714, 0.9082102461743181, 0.8327639909616653, 0.8412165129556433, 0.4501079848308764, 0.37910265316864683, 0.7962968948263066, 0.9637301587301587, 0.9478726401803326, 1.0, 0.9723622782446313, 0.6435338114701993, 0.7624308750035494, 0.24391154489459244, 0.8695966683099036, 0.9833333333333333, 0.8529166983114351, 0.6744159166966377, 0.7107669242284628, 0.9722222222222222, 0.34103286042596237, 0.48375258597384035, 0.4872811794555492, 0.9185131389078758, 0.5106795978730676, 0.8526183270920114, 0.4559803863586022, 0.5101836917800987, 0.9303860409742762, 0.5692290082924758, 0.6485977597450663, 0.6739001708475393, 0.882582335964689, 0.6581558997868989, 0.47946644058549004, 0.7914133617874026, 0.6839049472844055, 0.7577320827320827, 0.3653901556272792, 0.9577777777777778, 0.9888888888888889, 0.7594231978847362, 0.6337257109000791, 0.32839658803258753, 0.6916716902223055, 0.6511496058114481, 0.6815117818451392, 0.6778615637236326, 0.45804776049346785, 0.9877124183006536, 0.764770072006648, 0.7374204160137796, 0.31837611080786893])
    }

    hybrid_cropped_no_clahe = {
        'rank1': np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]),
        'rank5': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        'ap': np.array([0.7872452547452549, 0.1959469061113069, 0.4584363804044656, 0.679355491518963, 0.8452344236959622, 1.0, 0.6959773176767137, 0.40816191691305953, 0.9656926406926407, 0.11245876007923758, 0.8037716224274155, 0.2843201579019557, 0.20904388208888255, 0.7565709465419583, 0.7769828637219942, 0.779771043492848, 0.17664029407499207, 0.7329109961726963, 0.19291618655548662, 0.7934277808843554, 0.3677579365079365, 0.6005114805471113, 0.23079032939352048, 0.8431286894923258, 0.9707362082362082, 0.3436585939390892, 0.25649514797292056, 0.6981231502252774, 0.9707362082362082, 0.8832981028633201, 0.6375249606616068, 0.0588629400847402, 0.43504853912580854, 0.5760165156113228, 0.3179732464183909, 0.6464973224756125, 0.5758402480932026, 0.689565836109615, 0.08411933134227366, 0.6798678185168868, 0.15849461617005475, 0.69484758949262, 0.5543741047312476, 0.30395317867086913, 0.9526939348920772, 0.5960348361539513, 0.7842207792207792, 0.1446822173688359, 0.8385718275914356, 0.7897818016095326, 0.6239225589225588, 0.406708676168789, 0.514049498646542, 0.6438996180934435, 0.934915084915085, 0.8453839963455348, 0.9809090909090908, 0.9506919006919007, 0.6292944869529328, 0.9005494505494503, 0.21384514793901666, 0.7016343123251018, 0.8948963278476464, 0.8213729788729789, 0.5211288962647659, 0.6114903637856492, 0.8304945054945054, 0.1203493536952515, 0.421855317788011, 0.483621746813654, 0.7736020095844229, 0.34641131317723745, 0.7200078333848949, 0.41920504734427955, 0.5400601337561786, 0.8810838072602777, 0.5133319635276158, 0.6381957308445719, 0.6366922249125638, 0.636811633785318, 0.5413774424669129, 0.6771468701606996, 0.687499724371957, 0.5861739836070163, 0.31271845438512114, 0.4334847253268306, 0.9473767752715122, 0.9665384615384617, 0.6484413797125661, 0.6740330105536105, 0.343064487798089, 0.6037050637740292, 0.6822523357584334, 0.7640725551251867, 0.62845887762977, 0.39859663663445044, 0.939232697127434, 0.630947466230485, 0.7792760996647637, 0.2600552927283677])
    }

    hybrid_cropped_clahe = hybrid_cropped_metrics

    hypothesis_3(hybrid_uncropped_clahe, hybrid_uncropped_no_clahe, hybrid_cropped_clahe, hybrid_cropped_no_clahe)
