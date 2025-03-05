


def _tabular_begin(ncols):
  out = "\\" + "begin{tabular}{" + 'l' + 'c'*ncols + "}\n" \
    "\hline\hline\n"
  return out

def _tabular_end():
  out = "\hline\hline\n" + "\end{tabular}"
  return out

def _fill_tabular_row(key: str, values: list):
  out = ' ' + key
  for v in values:
    out += ' & ' + f'${v}$' 
  out += "  \\\\" + "\n"
  return out


def export_to_table(rows_dict, cols=None, save=None):
  len_data = len(list(rows_dict.values())[0])
  assert(len(cols)==len_data)

  tab = _tabular_begin(len_data)
  if cols is not None:
    for c in cols:
      tab += " & " + c
    tab += " \\\\" + "\n" \
           "\hline\hline\n"
  for k, v in rows_dict.items():
    tab += _fill_tabular_row(k, v)
  tab += _tabular_end()
  tab = tab.replace('+-', '\pm')

  if save is not None:
    with open(save, 'w') as of:
      of.write(tab)

  return tab


