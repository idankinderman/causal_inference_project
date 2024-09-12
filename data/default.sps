file handle pcdat/name='default.dat' /lrecl=98.
data list file pcdat free /
  R0000100 (F5)
  R0000600 (F2)
  R0001800 (F2)
  R0008800 (F2)
  R0009100 (F2)
  R0009600 (F2)
  R0010300 (F2)
  R0010400 (F2)
  R0010500 (F2)
  R0010600 (F2)
  R0013200 (F2)
  R0013400 (F2)
  R0015300 (F2)
  R0145100 (F2)
  R0146100 (F2)
  R0149800 (F2)
  R0149900 (F2)
  R0173600 (F2)
  R0214700 (F2)
  R0214800 (F2)
  R0216400 (F2)
  R0216601 (F2)
  R0217900 (F5)
  R0217910 (F2)
  T0857100 (F2)
  T2015100 (F2)
  T2998600 (F2)
  T3942500 (F2)
  T4876800 (F2)
  T5593000 (F2)
.
* The following code works with current versions of SPSS.
missing values all (-5 thru -1).
* older versions of SPSS may require this:
* recode all (-5,-3,-2,-1=-4).
* missing values all (-4).
variable labels
  R0000100  "ID# (1-12686) 79"
  R0000600  "AGE OF R 79"
  R0001800  "AREA RESIDENCE @ AGE 14 URBAN/RURAL 79"
  R0008800  "RS MOTHER AND FATHER LIVE IN SAME HH 79"
  R0009100  "# OF SIBS 79"
  R0009600  "1ST/ONLY RACL/ETHNIC ORIGIN 79"
  R0010300  "RELGN R RAISED COLLAPSED 79"
  R0010400  "PRSNT RELGS AFFILIATION 79"
  R0010500  "FREQ RELGS ATTENDANCE R 79"
  R0010600  "MARITAL STATUS 79"
  R0013200  "# CHILDREN IDEAL FOR FAMILY 79"
  R0013400  "# CHILDREN R HAD 79"
  R0015300  "TOT# CHILDREN EXPCT HAVE 79"
  R0145100  "ICHK HLTH LIMITATIONS I2A/2B YES"
  R0146100  "ICHK MAIN HLTH COND CAUSE WRK LIMITS 79"
  R0149800  "PERSON INFLUENCE R DEC NO CHILDREN 79"
  R0149900  "ATND R DEC PURSUE CAREER DELAY FAM 79"
  R0173600  "SAMPLE ID  79 INT"
  R0214700  "RACL/ETHNIC COHORT /SCRNR 79"
  R0214800  "SEX OF R 79"
  R0216400  "REGION OF CURRENT RESIDENCE 79"
  R0216601  "ENROLLMT STAT MAY 1 SURVEY YR (REV) 79"
  R0217900  "TOT NET FAMILY INC P-C YR 79"
  R0217910  "POVERTY STATUS 79"
  T0857100  "# BIO CHILDREN REPORTED 2006"
  T2015100  "# BIO CHILDREN REPORTED 2008"
  T2998600  "# BIO CHILDREN REPORTED 2010"
  T3942500  "# BIO CHILDREN REPORTED 2012"
  T4876800  "# BIO CHILDREN REPORTED 2014"
  T5593000  "# BIO CHILDREN REPORTED 2016"
.

* Recode continuous values. 
* recode 
 R0000600 
    (0 thru 13 eq 0)
    (14 thru 14 eq 14)
    (15 thru 15 eq 15)
    (16 thru 16 eq 16)
    (17 thru 17 eq 17)
    (18 thru 18 eq 18)
    (19 thru 19 eq 19)
    (20 thru 20 eq 20)
    (21 thru 21 eq 21)
    (22 thru 22 eq 22)
    (23 thru 23 eq 23)
    (24 thru 24 eq 24)
    (25 thru 25 eq 25)
    (26 thru 26 eq 26)
    (27 thru 27 eq 27)
    (28 thru 28 eq 28)
    (29 thru 29 eq 29)
    (30 thru 99999 eq 30)
    / 
 R0009100 
    (0 thru 0 eq 0)
    (1 thru 1 eq 1)
    (2 thru 2 eq 2)
    (3 thru 3 eq 3)
    (4 thru 4 eq 4)
    (5 thru 5 eq 5)
    (6 thru 6 eq 6)
    (7 thru 7 eq 7)
    (8 thru 8 eq 8)
    (9 thru 9 eq 9)
    (10 thru 10 eq 10)
    (11 thru 11 eq 11)
    (12 thru 12 eq 12)
    (13 thru 13 eq 13)
    (14 thru 14 eq 14)
    (15 thru 15 eq 15)
    (16 thru 99999 eq 16)
    / 
 R0013200 
    (0 thru 0 eq 0)
    (1 thru 1 eq 1)
    (2 thru 2 eq 2)
    (3 thru 3 eq 3)
    (4 thru 4 eq 4)
    (5 thru 5 eq 5)
    (6 thru 6 eq 6)
    (7 thru 7 eq 7)
    (8 thru 8 eq 8)
    (9 thru 9 eq 9)
    (10 thru 10 eq 10)
    (11 thru 11 eq 11)
    (12 thru 12 eq 12)
    (13 thru 13 eq 13)
    (14 thru 14 eq 14)
    (15 thru 15 eq 15)
    (16 thru 99999 eq 16)
    / 
 R0013400 
    (0 thru 0 eq 0)
    (1 thru 1 eq 1)
    (2 thru 2 eq 2)
    (3 thru 3 eq 3)
    (4 thru 4 eq 4)
    (5 thru 5 eq 5)
    (6 thru 6 eq 6)
    (7 thru 7 eq 7)
    (8 thru 8 eq 8)
    (9 thru 9 eq 9)
    (10 thru 10 eq 10)
    (11 thru 11 eq 11)
    (12 thru 12 eq 12)
    (13 thru 13 eq 13)
    (14 thru 14 eq 14)
    (15 thru 15 eq 15)
    (16 thru 99999 eq 16)
    / 
 R0015300 
    (0 thru 0 eq 0)
    (1 thru 1 eq 1)
    (2 thru 2 eq 2)
    (3 thru 3 eq 3)
    (4 thru 4 eq 4)
    (5 thru 5 eq 5)
    (6 thru 6 eq 6)
    (7 thru 7 eq 7)
    (8 thru 8 eq 8)
    (9 thru 9 eq 9)
    (10 thru 10 eq 10)
    (11 thru 11 eq 11)
    (12 thru 12 eq 12)
    (13 thru 13 eq 13)
    (14 thru 14 eq 14)
    (15 thru 15 eq 15)
    (16 thru 99999 eq 16)
    / 
 R0217900 
    (0 thru 0 eq 0)
    (1 thru 999 eq 1)
    (1000 thru 1999 eq 1000)
    (2000 thru 2999 eq 2000)
    (3000 thru 3999 eq 3000)
    (4000 thru 4999 eq 4000)
    (5000 thru 5999 eq 5000)
    (6000 thru 6999 eq 6000)
    (7000 thru 7999 eq 7000)
    (8000 thru 8999 eq 8000)
    (9000 thru 9999 eq 9000)
    (10000 thru 14999 eq 10000)
    (15000 thru 19999 eq 15000)
    (20000 thru 24999 eq 20000)
    (25000 thru 49999 eq 25000)
    (50000 thru 9999999 eq 50000)
    / 
 T0857100 
    (0 thru 0 eq 0)
    (1 thru 1 eq 1)
    (2 thru 2 eq 2)
    (3 thru 3 eq 3)
    (4 thru 4 eq 4)
    (5 thru 5 eq 5)
    (6 thru 6 eq 6)
    (7 thru 7 eq 7)
    (8 thru 8 eq 8)
    (9 thru 9 eq 9)
    (10 thru 999 eq 10)
    / 
 T2015100 
    (0 thru 0 eq 0)
    (1 thru 1 eq 1)
    (2 thru 2 eq 2)
    (3 thru 3 eq 3)
    (4 thru 4 eq 4)
    (5 thru 5 eq 5)
    (6 thru 6 eq 6)
    (7 thru 7 eq 7)
    (8 thru 8 eq 8)
    (9 thru 9 eq 9)
    (10 thru 999 eq 10)
    / 
 T2998600 
    (0 thru 0 eq 0)
    (1 thru 1 eq 1)
    (2 thru 2 eq 2)
    (3 thru 3 eq 3)
    (4 thru 4 eq 4)
    (5 thru 5 eq 5)
    (6 thru 6 eq 6)
    (7 thru 7 eq 7)
    (8 thru 8 eq 8)
    (9 thru 9 eq 9)
    (10 thru 999 eq 10)
    / 
 T3942500 
    (0 thru 0 eq 0)
    (1 thru 1 eq 1)
    (2 thru 2 eq 2)
    (3 thru 3 eq 3)
    (4 thru 4 eq 4)
    (5 thru 5 eq 5)
    (6 thru 6 eq 6)
    (7 thru 7 eq 7)
    (8 thru 8 eq 8)
    (9 thru 9 eq 9)
    (10 thru 999 eq 10)
    / 
 T4876800 
    (0 thru 0 eq 0)
    (1 thru 1 eq 1)
    (2 thru 2 eq 2)
    (3 thru 3 eq 3)
    (4 thru 4 eq 4)
    (5 thru 5 eq 5)
    (6 thru 6 eq 6)
    (7 thru 7 eq 7)
    (8 thru 8 eq 8)
    (9 thru 9 eq 9)
    (10 thru 999 eq 10)
    / 
 T5593000 
    (0 thru 0 eq 0)
    (1 thru 1 eq 1)
    (2 thru 2 eq 2)
    (3 thru 3 eq 3)
    (4 thru 4 eq 4)
    (5 thru 5 eq 5)
    (6 thru 6 eq 6)
    (7 thru 7 eq 7)
    (8 thru 8 eq 8)
    (9 thru 9 eq 9)
    (10 thru 999 eq 10)
.

* value labels
 R0000600
    0 "0 TO 13: < 14"
    14 "14"
    15 "15"
    16 "16"
    17 "17"
    18 "18"
    19 "19"
    20 "20"
    21 "21"
    22 "22"
    23 "23"
    24 "24"
    25 "25"
    26 "26"
    27 "27"
    28 "28"
    29 "29"
    30 "30 TO 99999: 30+"
    /
 R0001800
    1 "IN TOWN OR CITY"
    2 "IN COUNTRY-NOT FARM"
    3 "ON FARM OR RANCH"
    /
 R0008800
    1 "YES"
    0 "NO"
    /
 R0009100
    0 "0"
    1 "1"
    2 "2"
    3 "3"
    4 "4"
    5 "5"
    6 "6"
    7 "7"
    8 "8"
    9 "9"
    10 "10"
    11 "11"
    12 "12"
    13 "13"
    14 "14"
    15 "15"
    16 "16 TO 99999: 16+"
    /
 R0009600
    0 "NONE"
    1 "BLACK"
    2 "CHINESE"
    3 "ENGLISH"
    4 "FILIPINO"
    5 "FRENCH"
    6 "GERMAN"
    7 "GREEK"
    8 "HAWAIIAN, P.I."
    9 "INDIAN-AMERICAN OR NATIVE AMERICAN"
    10 "ASIAN INDIAN"
    11 "IRISH"
    12 "ITALIAN"
    13 "JAPANESE"
    14 "KOREAN"
    15 "CUBAN"
    16 "CHICANO"
    17 "MEXICAN"
    18 "MEXICAN-AMER"
    19 "PUERTO RICAN"
    20 "OTHER HISPANIC"
    21 "OTHER SPANISH"
    22 "POLISH"
    23 "PORTUGUESE"
    24 "RUSSIAN"
    25 "SCOTTISH"
    26 "VIETNAMESE"
    27 "WELSH"
    28 "OTHER"
    29 "AMERICAN"
    /
 R0010300
    0 "NONE, NO RELIGION"
    1 "PROTESTANT"
    2 "BAPTIST"
    3 "EPISCOPALIAN"
    4 "LUTHERAN"
    5 "METHODIST"
    6 "PRESBYTERIAN"
    7 "ROMAN CATHOLIC"
    8 "JEWISH"
    9 "OTHER"
    /
 R0010400
    0 "NONE, NO RELIGION"
    1 "PROTESTANT"
    2 "BAPTIST"
    3 "EPISCOPALIAN"
    4 "LUTHERAN"
    5 "METHODIST"
    6 "PRESBYTERIAN"
    7 "ROMAN CATHOLIC"
    8 "JEWISH"
    9 "OTHER"
    /
 R0010500
    1 "NOT AT ALL"
    2 "INFREQUENTLY"
    3 "ONCE PER MONTH"
    4 "2-3 TIMES PER MONTH"
    5 "ONCE PER WEEK"
    6 "> ONCE PER WEEK"
    /
 R0010600
    1 "PRESENTLY MARRIED"
    2 "WIDOWED"
    3 "DIVORCED"
    4 "SEPARATED"
    5 "NEVER MARRIED-ANNUL"
    /
 R0013200
    0 "0"
    1 "1"
    2 "2"
    3 "3"
    4 "4"
    5 "5"
    6 "6"
    7 "7"
    8 "8"
    9 "9"
    10 "10"
    11 "11"
    12 "12"
    13 "13"
    14 "14"
    15 "15"
    16 "16 TO 99999: 16+"
    /
 R0013400
    0 "0"
    1 "1"
    2 "2"
    3 "3"
    4 "4"
    5 "5"
    6 "6"
    7 "7"
    8 "8"
    9 "9"
    10 "10"
    11 "11"
    12 "12"
    13 "13"
    14 "14"
    15 "15"
    16 "16 TO 99999: 16+"
    /
 R0015300
    0 "0"
    1 "1"
    2 "2"
    3 "3"
    4 "4"
    5 "5"
    6 "6"
    7 "7"
    8 "8"
    9 "9"
    10 "10"
    11 "11"
    12 "12"
    13 "13"
    14 "14"
    15 "15"
    16 "16 TO 99999: 16+"
    /
 R0145100
    1 "YES"
    0 "NO"
    /
 R0146100
    1 "ACCIDENT-INJURY"
    2 "IN BOX B"
    3 "NEITHER"
    4 "NORMAL PREGNANCY"
    5 "NORMAL DELIVERY"
    6 "VASECTOMY-T.L."
    /
 R0149800
    1 "STRONGLY DISAPPROVE"
    2 "SOMEWHAT DISAPPROVE"
    3 "SOMEWHAT APPROVE"
    4 "STRONGLY APPROVE"
    /
 R0149900
    1 "STRONGLY DISAPPROVE"
    2 "SOMEWHAT DISAPPROVE"
    3 "SOMEWHAT APPROVE"
    4 "STRONGLY APPROVE"
    /
 R0173600
    1 "CROSS MALE WHITE"
    2 "CROSS MALE WH. POOR"
    3 "CROSS MALE BLACK"
    4 "CROSS MALE HISPANIC"
    5 "CROSS FEMALE WHITE"
    6 "CROSS FEMALE WH POOR"
    7 "CROSS FEMALE BLACK"
    8 "CROSS FEMALE HISPANIC"
    9 "SUP MALE WH POOR"
    10 "SUP MALE BLACK"
    11 "SUP MALE HISPANIC"
    12 "SUP FEM WH POOR"
    13 "SUP FEMALE BLACK"
    14 "SUP FEMALE HISPANIC"
    15 "MIL MALE WHITE"
    16 "MIL MALE BLACK"
    17 "MIL MALE HISPANIC"
    18 "MIL FEMALE WHITE"
    19 "MIL FEMALE BLACK"
    20 "MIL FEMALE HISPANIC"
    /
 R0214700
    1 "HISPANIC"
    2 "BLACK"
    3 "NON-BLACK, NON-HISPANIC"
    /
 R0214800
    1 "MALE"
    2 "FEMALE"
    /
 R0216400
    1 "NORTHEAST"
    2 "NORTH CENTRAL"
    3 "SOUTH"
    4 "WEST"
    /
 R0216601
    1 "NOT ENROLLED, COMPLETED LESS THAN 12TH GRADE"
    2 "ENROLLED IN HIGH SCHOOL"
    3 "ENROLLED IN COLLEGE"
    4 "NOT ENROLLED, HIGH SCHOOL GRADUATE"
    /
 R0217900
    0 "0"
    1 "1 TO 999"
    1000 "1000 TO 1999"
    2000 "2000 TO 2999"
    3000 "3000 TO 3999"
    4000 "4000 TO 4999"
    5000 "5000 TO 5999"
    6000 "6000 TO 6999"
    7000 "7000 TO 7999"
    8000 "8000 TO 8999"
    9000 "9000 TO 9999"
    10000 "10000 TO 14999"
    15000 "15000 TO 19999"
    20000 "20000 TO 24999"
    25000 "25000 TO 49999"
    50000 "50000 TO 9999999: 50000+"
    /
 R0217910
    1 "IN POVERTY"
    0 "NOT IN POVERTY"
    /
 T0857100
    0 "0"
    1 "1"
    2 "2"
    3 "3"
    4 "4"
    5 "5"
    6 "6"
    7 "7"
    8 "8"
    9 "9"
    10 "10 TO 999: 10+"
    /
 T2015100
    0 "0"
    1 "1"
    2 "2"
    3 "3"
    4 "4"
    5 "5"
    6 "6"
    7 "7"
    8 "8"
    9 "9"
    10 "10 TO 999: 10+"
    /
 T2998600
    0 "0"
    1 "1"
    2 "2"
    3 "3"
    4 "4"
    5 "5"
    6 "6"
    7 "7"
    8 "8"
    9 "9"
    10 "10 TO 999: 10+"
    /
 T3942500
    0 "0"
    1 "1"
    2 "2"
    3 "3"
    4 "4"
    5 "5"
    6 "6"
    7 "7"
    8 "8"
    9 "9"
    10 "10 TO 999: 10+"
    /
 T4876800
    0 "0"
    1 "1"
    2 "2"
    3 "3"
    4 "4"
    5 "5"
    6 "6"
    7 "7"
    8 "8"
    9 "9"
    10 "10 TO 999: 10+"
    /
 T5593000
    0 "0"
    1 "1"
    2 "2"
    3 "3"
    4 "4"
    5 "5"
    6 "6"
    7 "7"
    8 "8"
    9 "9"
    10 "10 TO 999: 10+"
    /
.
/* Crosswalk for Reference number & Question name
 * Uncomment and edit this RENAME VARIABLES statement to rename variables for ease of use.
 * This command does not guarantee uniqueness
 */  /* *start* */

* RENAME VARIABLES
  (R0000100 = CASEID_1979) 
  (R0000600 = FAM_1B_1979)   /* FAM-1B */
  (R0001800 = FAM_6_1979)   /* FAM-6 */
  (R0008800 = FAM_27C_1979)   /* FAM-27C */
  (R0009100 = FAM_28A_1979)   /* FAM-28A */
  (R0009600 = FAM_30_1_1979)   /* FAM-30_1 */
  (R0010300 = R_REL_1_COL_1979)   /* R_REL-1_COL */
  (R0010400 = R_REL_2_COL_1979)   /* R_REL-2_COL */
  (R0010500 = R_REL_3_1979)   /* R_REL-3 */
  (R0010600 = MAR_1_1979)   /* MAR-1 */
  (R0013200 = FER_1B_1979)   /* FER-1B */
  (R0013400 = FER_2A_1979)   /* FER-2A */
  (R0015300 = FER_3_1979)   /* FER-3 */
  (R0145100 = Q11_5A_1979)   /* Q11-5A */
  (R0146100 = HEALTH_8A_1979)   /* HEALTH-8A */
  (R0149800 = OTHER_3G_1979)   /* OTHER-3G */
  (R0149900 = OTHER_3H_1979)   /* OTHER-3H */
  (R0173600 = SAMPLE_ID_1979) 
  (R0214700 = SAMPLE_RACE_78SCRN) 
  (R0214800 = SAMPLE_SEX_1979) 
  (R0216400 = REGION_1979) 
  (R0216601 = ENROLLMTREV79_1979) 
  (R0217900 = TNFI_TRUNC_1979) 
  (R0217910 = POVSTATUS_1979) 
  (T0857100 = Q10_2_2006)   /* Q10-2 */
  (T2015100 = Q10_2_2008)   /* Q10-2 */
  (T2998600 = Q10_2_2010)   /* Q10-2 */
  (T3942500 = Q10_2_2012)   /* Q10-2 */
  (T4876800 = Q10_2_2014)   /* Q10-2 */
  (T5593000 = Q10_2_2016)   /* Q10-2 */
.
  /* *end* */

descriptives all.

*--- Tabulations using reference number variables.
*freq var=R0000100, 
  R0000600, 
  R0001800, 
  R0008800, 
  R0009100, 
  R0009600, 
  R0010300, 
  R0010400, 
  R0010500, 
  R0010600, 
  R0013200, 
  R0013400, 
  R0015300, 
  R0145100, 
  R0146100, 
  R0149800, 
  R0149900, 
  R0173600, 
  R0214700, 
  R0214800, 
  R0216400, 
  R0216601, 
  R0217900, 
  R0217910, 
  T0857100, 
  T2015100, 
  T2998600, 
  T3942500, 
  T4876800, 
  T5593000.

*--- Tabulations using qname variables.
*freq var=CASEID_1979, 
  FAM_1B_1979, 
  FAM_6_1979, 
  FAM_27C_1979, 
  FAM_28A_1979, 
  FAM_30_1_1979, 
  R_REL_1_COL_1979, 
  R_REL_2_COL_1979, 
  R_REL_3_1979, 
  MAR_1_1979, 
  FER_1B_1979, 
  FER_2A_1979, 
  FER_3_1979, 
  Q11_5A_1979, 
  HEALTH_8A_1979, 
  OTHER_3G_1979, 
  OTHER_3H_1979, 
  SAMPLE_ID_1979, 
  SAMPLE_RACE_78SCRN, 
  SAMPLE_SEX_1979, 
  REGION_1979, 
  ENROLLMTREV79_1979, 
  TNFI_TRUNC_1979, 
  POVSTATUS_1979, 
  Q10_2_2006, 
  Q10_2_2008, 
  Q10_2_2010, 
  Q10_2_2012, 
  Q10_2_2014, 
  Q10_2_2016.
