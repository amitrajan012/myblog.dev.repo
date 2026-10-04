+++
title = "{{ replace .File.ContentBaseName "-" " " | title }}"
date = {{ .Date }}
draft = true
# kind: "concept" (one idea explained on its own) or "excerpt" (a section from the manuscript)
kind = "concept"
subtitle = "One sentence: the question this post answers."
# Optional — link the post to its place in the book:
# chapter = 6
# section = "6.5"
tags = []
+++

{{< problem >}}
The problem this idea exists to solve — before naming the method.
{{< /problem >}}

Opening paragraph.

{{< pullquote >}}The one sentence a reader should remember.{{< /pullquote >}}

{{< hood title="What the maths shows" >}}
Lead-in sentence.

$$ \theta \leftarrow \theta - \eta \nabla L(\theta) $$

Read the equation back in plain words.
{{< /hood >}}
