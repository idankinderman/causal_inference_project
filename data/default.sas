options nocenter validvarname=any;

*---Read in space-delimited ascii file;

data new_data;


infile 'default.dat' lrecl=98 missover DSD DLM=' ' print;
input
  R0000100
  R0000600
  R0001800
  R0008800
  R0009100
  R0009600
  R0010300
  R0010400
  R0010500
  R0010600
  R0013200
  R0013400
  R0015300
  R0145100
  R0146100
  R0149800
  R0149900
  R0173600
  R0214700
  R0214800
  R0216400
  R0216601
  R0217900
  R0217910
  T0857100
  T2015100
  T2998600
  T3942500
  T4876800
  T5593000
;
array nvarlist _numeric_;


*---Recode missing values to SAS custom system missing. See SAS
      documentation for use of MISSING option in procedures, e.g. PROC FREQ;

do over nvarlist;
  if nvarlist = -1 then nvarlist = .R;  /* Refused */
  if nvarlist = -2 then nvarlist = .D;  /* Dont know */
  if nvarlist = -3 then nvarlist = .I;  /* Invalid missing */
  if nvarlist = -4 then nvarlist = .V;  /* Valid missing */
  if nvarlist = -5 then nvarlist = .N;  /* Non-interview */
end;

  label R0000100 = "ID# (1-12686) 79";
  label R0000600 = "AGE OF R 79";
  label R0001800 = "AREA RESIDENCE @ AGE 14 URBAN/RURAL 79";
  label R0008800 = "RS MOTHER AND FATHER LIVE IN SAME HH 79";
  label R0009100 = "# OF SIBS 79";
  label R0009600 = "1ST/ONLY RACL/ETHNIC ORIGIN 79";
  label R0010300 = "RELGN R RAISED COLLAPSED 79";
  label R0010400 = "PRSNT RELGS AFFILIATION 79";
  label R0010500 = "FREQ RELGS ATTENDANCE R 79";
  label R0010600 = "MARITAL STATUS 79";
  label R0013200 = "# CHILDREN IDEAL FOR FAMILY 79";
  label R0013400 = "# CHILDREN R HAD 79";
  label R0015300 = "TOT# CHILDREN EXPCT HAVE 79";
  label R0145100 = "ICHK HLTH LIMITATIONS I2A/2B YES";
  label R0146100 = "ICHK MAIN HLTH COND CAUSE WRK LIMITS 79";
  label R0149800 = "PERSON INFLUENCE R DEC NO CHILDREN 79";
  label R0149900 = "ATND R DEC PURSUE CAREER DELAY FAM 79";
  label R0173600 = "SAMPLE ID  79 INT";
  label R0214700 = "RACL/ETHNIC COHORT /SCRNR 79";
  label R0214800 = "SEX OF R 79";
  label R0216400 = "REGION OF CURRENT RESIDENCE 79";
  label R0216601 = "ENROLLMT STAT MAY 1 SURVEY YR (REV) 79";
  label R0217900 = "TOT NET FAMILY INC P-C YR 79";
  label R0217910 = "POVERTY STATUS 79";
  label T0857100 = "# BIO CHILDREN REPORTED 2006";
  label T2015100 = "# BIO CHILDREN REPORTED 2008";
  label T2998600 = "# BIO CHILDREN REPORTED 2010";
  label T3942500 = "# BIO CHILDREN REPORTED 2012";
  label T4876800 = "# BIO CHILDREN REPORTED 2014";
  label T5593000 = "# BIO CHILDREN REPORTED 2016";

/*---------------------------------------------------------------------*
 *  Crosswalk for Reference number & Question name                     *
 *---------------------------------------------------------------------*
 * Uncomment and edit this RENAME statement to rename variables
 * for ease of use.  You may need to use  name literal strings
 * e.g.  'variable-name'n   to create valid SAS variable names, or 
 * alter variables similarly named across years.
 * This command does not guarantee uniqueness

 * See SAS documentation for use of name literals and use of the
 * VALIDVARNAME=ANY option.     
 *---------------------------------------------------------------------*/
  /* *start* */

* RENAME
  R0000100 = 'CASEID_1979'n
  R0000600 = 'FAM-1B_1979'n
  R0001800 = 'FAM-6_1979'n
  R0008800 = 'FAM-27C_1979'n
  R0009100 = 'FAM-28A_1979'n
  R0009600 = 'FAM-30_1_1979'n
  R0010300 = 'R_REL-1_COL_1979'n
  R0010400 = 'R_REL-2_COL_1979'n
  R0010500 = 'R_REL-3_1979'n
  R0010600 = 'MAR-1_1979'n
  R0013200 = 'FER-1B_1979'n
  R0013400 = 'FER-2A_1979'n
  R0015300 = 'FER-3_1979'n
  R0145100 = 'Q11-5A_1979'n
  R0146100 = 'HEALTH-8A_1979'n
  R0149800 = 'OTHER-3G_1979'n
  R0149900 = 'OTHER-3H_1979'n
  R0173600 = 'SAMPLE_ID_1979'n
  R0214700 = 'SAMPLE_RACE_78SCRN'n
  R0214800 = 'SAMPLE_SEX_1979'n
  R0216400 = 'REGION_1979'n
  R0216601 = 'ENROLLMTREV79_1979'n
  R0217900 = 'TNFI_TRUNC_1979'n
  R0217910 = 'POVSTATUS_1979'n
  T0857100 = 'Q10-2_2006'n
  T2015100 = 'Q10-2_2008'n
  T2998600 = 'Q10-2_2010'n
  T3942500 = 'Q10-2_2012'n
  T4876800 = 'Q10-2_2014'n
  T5593000 = 'Q10-2_2016'n
;
  /* *finish* */

run;

proc means data=new_data n mean min max;
run;


/*---------------------------------------------------------------------*
 *  FORMATTED TABULATIONS                                              *
 *---------------------------------------------------------------------*
 * You can uncomment and edit the PROC FORMAT and PROC FREQ statements 
 * provided below to obtain formatted tabulations. The tabulations 
 * should reflect codebook values.
 * 
 * Please edit the formats below reflect any renaming of the variables
 * you may have done in the first data step. 
 *---------------------------------------------------------------------*/

/*
proc format; 
value vx1f
  0-13='0 TO 13: < 14'
  14='14'
  15='15'
  16='16'
  17='17'
  18='18'
  19='19'
  20='20'
  21='21'
  22='22'
  23='23'
  24='24'
  25='25'
  26='26'
  27='27'
  28='28'
  29='29'
  30-99999='30 TO 99999: 30+'
;
value vx2f
  1='IN TOWN OR CITY'
  2='IN COUNTRY-NOT FARM'
  3='ON FARM OR RANCH'
;
value vx3f
  1='YES'
  0='NO'
;
value vx4f
  0='0'
  1='1'
  2='2'
  3='3'
  4='4'
  5='5'
  6='6'
  7='7'
  8='8'
  9='9'
  10='10'
  11='11'
  12='12'
  13='13'
  14='14'
  15='15'
  16-99999='16 TO 99999: 16+'
;
value vx5f
  0='NONE'
  1='BLACK'
  2='CHINESE'
  3='ENGLISH'
  4='FILIPINO'
  5='FRENCH'
  6='GERMAN'
  7='GREEK'
  8='HAWAIIAN, P.I.'
  9='INDIAN-AMERICAN OR NATIVE AMERICAN'
  10='ASIAN INDIAN'
  11='IRISH'
  12='ITALIAN'
  13='JAPANESE'
  14='KOREAN'
  15='CUBAN'
  16='CHICANO'
  17='MEXICAN'
  18='MEXICAN-AMER'
  19='PUERTO RICAN'
  20='OTHER HISPANIC'
  21='OTHER SPANISH'
  22='POLISH'
  23='PORTUGUESE'
  24='RUSSIAN'
  25='SCOTTISH'
  26='VIETNAMESE'
  27='WELSH'
  28='OTHER'
  29='AMERICAN'
;
value vx6f
  0='NONE, NO RELIGION'
  1='PROTESTANT'
  2='BAPTIST'
  3='EPISCOPALIAN'
  4='LUTHERAN'
  5='METHODIST'
  6='PRESBYTERIAN'
  7='ROMAN CATHOLIC'
  8='JEWISH'
  9='OTHER'
;
value vx7f
  0='NONE, NO RELIGION'
  1='PROTESTANT'
  2='BAPTIST'
  3='EPISCOPALIAN'
  4='LUTHERAN'
  5='METHODIST'
  6='PRESBYTERIAN'
  7='ROMAN CATHOLIC'
  8='JEWISH'
  9='OTHER'
;
value vx8f
  1='NOT AT ALL'
  2='INFREQUENTLY'
  3='ONCE PER MONTH'
  4='2-3 TIMES PER MONTH'
  5='ONCE PER WEEK'
  6='> ONCE PER WEEK'
;
value vx9f
  1='PRESENTLY MARRIED'
  2='WIDOWED'
  3='DIVORCED'
  4='SEPARATED'
  5='NEVER MARRIED-ANNUL'
;
value vx10f
  0='0'
  1='1'
  2='2'
  3='3'
  4='4'
  5='5'
  6='6'
  7='7'
  8='8'
  9='9'
  10='10'
  11='11'
  12='12'
  13='13'
  14='14'
  15='15'
  16-99999='16 TO 99999: 16+'
;
value vx11f
  0='0'
  1='1'
  2='2'
  3='3'
  4='4'
  5='5'
  6='6'
  7='7'
  8='8'
  9='9'
  10='10'
  11='11'
  12='12'
  13='13'
  14='14'
  15='15'
  16-99999='16 TO 99999: 16+'
;
value vx12f
  0='0'
  1='1'
  2='2'
  3='3'
  4='4'
  5='5'
  6='6'
  7='7'
  8='8'
  9='9'
  10='10'
  11='11'
  12='12'
  13='13'
  14='14'
  15='15'
  16-99999='16 TO 99999: 16+'
;
value vx13f
  1='YES'
  0='NO'
;
value vx14f
  1='ACCIDENT-INJURY'
  2='IN BOX B'
  3='NEITHER'
  4='NORMAL PREGNANCY'
  5='NORMAL DELIVERY'
  6='VASECTOMY-T.L.'
;
value vx15f
  1='STRONGLY DISAPPROVE'
  2='SOMEWHAT DISAPPROVE'
  3='SOMEWHAT APPROVE'
  4='STRONGLY APPROVE'
;
value vx16f
  1='STRONGLY DISAPPROVE'
  2='SOMEWHAT DISAPPROVE'
  3='SOMEWHAT APPROVE'
  4='STRONGLY APPROVE'
;
value vx17f
  1='CROSS MALE WHITE'
  2='CROSS MALE WH. POOR'
  3='CROSS MALE BLACK'
  4='CROSS MALE HISPANIC'
  5='CROSS FEMALE WHITE'
  6='CROSS FEMALE WH POOR'
  7='CROSS FEMALE BLACK'
  8='CROSS FEMALE HISPANIC'
  9='SUP MALE WH POOR'
  10='SUP MALE BLACK'
  11='SUP MALE HISPANIC'
  12='SUP FEM WH POOR'
  13='SUP FEMALE BLACK'
  14='SUP FEMALE HISPANIC'
  15='MIL MALE WHITE'
  16='MIL MALE BLACK'
  17='MIL MALE HISPANIC'
  18='MIL FEMALE WHITE'
  19='MIL FEMALE BLACK'
  20='MIL FEMALE HISPANIC'
;
value vx18f
  1='HISPANIC'
  2='BLACK'
  3='NON-BLACK, NON-HISPANIC'
;
value vx19f
  1='MALE'
  2='FEMALE'
;
value vx20f
  1='NORTHEAST'
  2='NORTH CENTRAL'
  3='SOUTH'
  4='WEST'
;
value vx21f
  1='NOT ENROLLED, COMPLETED LESS THAN 12TH GRADE'
  2='ENROLLED IN HIGH SCHOOL'
  3='ENROLLED IN COLLEGE'
  4='NOT ENROLLED, HIGH SCHOOL GRADUATE'
;
value vx22f
  0='0'
  1-999='1 TO 999'
  1000-1999='1000 TO 1999'
  2000-2999='2000 TO 2999'
  3000-3999='3000 TO 3999'
  4000-4999='4000 TO 4999'
  5000-5999='5000 TO 5999'
  6000-6999='6000 TO 6999'
  7000-7999='7000 TO 7999'
  8000-8999='8000 TO 8999'
  9000-9999='9000 TO 9999'
  10000-14999='10000 TO 14999'
  15000-19999='15000 TO 19999'
  20000-24999='20000 TO 24999'
  25000-49999='25000 TO 49999'
  50000-9999999='50000 TO 9999999: 50000+'
;
value vx23f
  1='IN POVERTY'
  0='NOT IN POVERTY'
;
value vx24f
  0='0'
  1='1'
  2='2'
  3='3'
  4='4'
  5='5'
  6='6'
  7='7'
  8='8'
  9='9'
  10-999='10 TO 999: 10+'
;
value vx25f
  0='0'
  1='1'
  2='2'
  3='3'
  4='4'
  5='5'
  6='6'
  7='7'
  8='8'
  9='9'
  10-999='10 TO 999: 10+'
;
value vx26f
  0='0'
  1='1'
  2='2'
  3='3'
  4='4'
  5='5'
  6='6'
  7='7'
  8='8'
  9='9'
  10-999='10 TO 999: 10+'
;
value vx27f
  0='0'
  1='1'
  2='2'
  3='3'
  4='4'
  5='5'
  6='6'
  7='7'
  8='8'
  9='9'
  10-999='10 TO 999: 10+'
;
value vx28f
  0='0'
  1='1'
  2='2'
  3='3'
  4='4'
  5='5'
  6='6'
  7='7'
  8='8'
  9='9'
  10-999='10 TO 999: 10+'
;
value vx29f
  0='0'
  1='1'
  2='2'
  3='3'
  4='4'
  5='5'
  6='6'
  7='7'
  8='8'
  9='9'
  10-999='10 TO 999: 10+'
;
*/

/* 
 *--- Tabulations using reference number variables;
proc freq data=new_data;
tables _ALL_ /MISSING;
  format R0000600 vx1f.;
  format R0001800 vx2f.;
  format R0008800 vx3f.;
  format R0009100 vx4f.;
  format R0009600 vx5f.;
  format R0010300 vx6f.;
  format R0010400 vx7f.;
  format R0010500 vx8f.;
  format R0010600 vx9f.;
  format R0013200 vx10f.;
  format R0013400 vx11f.;
  format R0015300 vx12f.;
  format R0145100 vx13f.;
  format R0146100 vx14f.;
  format R0149800 vx15f.;
  format R0149900 vx16f.;
  format R0173600 vx17f.;
  format R0214700 vx18f.;
  format R0214800 vx19f.;
  format R0216400 vx20f.;
  format R0216601 vx21f.;
  format R0217900 vx22f.;
  format R0217910 vx23f.;
  format T0857100 vx24f.;
  format T2015100 vx25f.;
  format T2998600 vx26f.;
  format T3942500 vx27f.;
  format T4876800 vx28f.;
  format T5593000 vx29f.;
run;
*/

/*
*--- Tabulations using default named variables;
proc freq data=new_data;
tables _ALL_ /MISSING;
  format 'FAM-1B_1979'n vx1f.;
  format 'FAM-6_1979'n vx2f.;
  format 'FAM-27C_1979'n vx3f.;
  format 'FAM-28A_1979'n vx4f.;
  format 'FAM-30_1_1979'n vx5f.;
  format 'R_REL-1_COL_1979'n vx6f.;
  format 'R_REL-2_COL_1979'n vx7f.;
  format 'R_REL-3_1979'n vx8f.;
  format 'MAR-1_1979'n vx9f.;
  format 'FER-1B_1979'n vx10f.;
  format 'FER-2A_1979'n vx11f.;
  format 'FER-3_1979'n vx12f.;
  format 'Q11-5A_1979'n vx13f.;
  format 'HEALTH-8A_1979'n vx14f.;
  format 'OTHER-3G_1979'n vx15f.;
  format 'OTHER-3H_1979'n vx16f.;
  format 'SAMPLE_ID_1979'n vx17f.;
  format 'SAMPLE_RACE_78SCRN'n vx18f.;
  format 'SAMPLE_SEX_1979'n vx19f.;
  format 'REGION_1979'n vx20f.;
  format 'ENROLLMTREV79_1979'n vx21f.;
  format 'TNFI_TRUNC_1979'n vx22f.;
  format 'POVSTATUS_1979'n vx23f.;
  format 'Q10-2_2006'n vx24f.;
  format 'Q10-2_2008'n vx25f.;
  format 'Q10-2_2010'n vx26f.;
  format 'Q10-2_2012'n vx27f.;
  format 'Q10-2_2014'n vx28f.;
  format 'Q10-2_2016'n vx29f.;
run;
*/