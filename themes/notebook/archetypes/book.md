+++
title = "{{ replace .File.ContentBaseName "-" " " | title }}"
date = {{ .Date }}
draft = true
part = 1
chapter = 1
section = "1.1"
weight = 11
subtitle = "One sentence: the question this section answers."
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
