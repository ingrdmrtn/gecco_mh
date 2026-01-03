import pandas as pd
from typing import Optional
import string

def narrative(
    df: pd.DataFrame,
    template: str,
    id_col: str = "participant",
    trial_col: str = "trial",
    max_trials: Optional[int] = None,
):
    """
    Generate narrative text from behavioral data using a configurable template.

    Args:
        df (pd.DataFrame): The dataset (must include id_col and trial_col).
        template (str): Template string with placeholders like {choice_1}, {reward}.
        id_col (str): Participant identifier column.
        trial_col (str): Trial index column.
        max_trials (int | None): Maximum number of trials to include per participant.
    Returns:
        str: Human-readable narrative text.
    """
    placeholders = [f[1] for f in string.Formatter().parse(template) if f[1] is not None]
    missing = [p for p in placeholders if p not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    text_blocks = []
    for pid, group in df.groupby(id_col):
        if max_trials:
            group = group.head(max_trials)

        participant_text = [f"Here is data from participant {pid}:"]
        for _, row in group.iterrows():
            row_dict = {key: row[key] for key in placeholders}
            trial = int(row[trial_col]) if trial_col in row else _
            trial_text = template.format(**row_dict)
            participant_text.append(f"\n    Trial {trial}:\n    {trial_text}")
        text_blocks.append("\n".join(participant_text))

    return "\n\n".join(text_blocks)


def get_data2text_function(name):
    if name == "narrative":
        def data2text(df, id_col, template, fit_type,
                      metadata=None, max_trials=None, max_blocks=None,
                      value_mappings=None):

            narratives = []
            has_blocks = "blocks" in df.columns  # ✅ FIXED

            for pid in df[id_col].unique():
                sub = df[df[id_col] == pid]
                trial_lines = []

                if has_blocks and max_trials is not None:
                    grouped = sub.groupby("blocks", sort=True)

                    # ✅ LIMIT NUMBER OF BLOCKS
                    for i, (block_id, block_df) in enumerate(grouped):
                        if max_blocks is not None and i >= max_blocks:
                            break

                        block_df = block_df.head(max_trials)

                        for _, row in block_df.iterrows():
                            vals = dict(row)

                            if value_mappings:
                                if not isinstance(value_mappings, dict):
                                    value_mappings = vars(value_mappings)

                                for col, mapping in value_mappings.items():
                                    if not isinstance(mapping, dict):
                                        mapping = vars(mapping)
                                    if col in vals:
                                        key = (
                                            str(int(vals[col]))
                                            if isinstance(vals[col], (int, float))
                                            else str(vals[col])
                                        )
                                        vals[col] = mapping.get(key, vals[col])

                            try:
                                trial_lines.append(template.format(**vals))
                            except KeyError as e:
                                print(
                                    f"[⚠️ GeCCo] Missing column {e} in template "
                                    f"for participant {pid}, block {block_id}"
                                )

                else:
                    # Fallback behavior
                    sub = sub.head(max_trials or len(sub))
                    for _, row in sub.iterrows():
                        vals = dict(row)
                        try:
                            trial_lines.append(template.format(**vals))
                        except KeyError:
                            continue

                participant_text = "\n".join(trial_lines)

                if fit_type == "individual":
                    narratives.append(f"Participant data:\n{participant_text}\n")
                else:
                    narratives.append(f"Participant {pid}:\n{participant_text}\n")

            return "\n".join(narratives)

        return data2text
