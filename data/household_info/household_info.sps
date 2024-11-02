file handle pcdat/name='household_info.dat' /lrecl=14.
data list file pcdat free /
  R0000100 (F5)
  R0000149 (F5)
.
* The following code works with current versions of SPSS.
missing values all (-5 thru -1).
* older versions of SPSS may require this:
* recode all (-5,-3,-2,-1=-4).
* missing values all (-4).
variable labels
  R0000100  "ID# (1-12686) 79"
  R0000149  "HH ID # 79"
.

* Recode continuous values. 
* recode 
.

* value labels
.
/* Crosswalk for Reference number & Question name
 * Uncomment and edit this RENAME VARIABLES statement to rename variables for ease of use.
 * This command does not guarantee uniqueness
 */  /* *start* */

* RENAME VARIABLES
  (R0000100 = CASEID_1979) 
  (R0000149 = HHID_1979) 
.
  /* *end* */

descriptives all.

*--- Tabulations using reference number variables.
*freq var=R0000100, 
  R0000149.

*--- Tabulations using qname variables.
*freq var=CASEID_1979, 
  HHID_1979.
