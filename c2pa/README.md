# Using TrustMark with C2PA

Open standards such as Content Credentials, developed by the [Coalition for Content Provenance and Authenticity(C2PA)](https://c2pa.org/), describe ways to encode information about an image’s history or _provenance_, such as how and when it was made. This information is usually carried within the image’s metadata.

## Durable Content Credentials

C2PA manifest data can be accidentally removed when the image is shared through platforms that do not yet support the standard. If a copy of the manifest data is retained in a database, the TrustMark identifier carried inside the watermark can be used as a key to look up that information from the database. This is referred to as a [_Durable Content Credential_](https://contentauthenticity.org/blog/durable-content-credentials) and the technical term for the identifier is a _soft binding_.

To create a soft binding, TrustMark encodes a random identifier via one of the encoding types in the [data schema](../README.md#data-schema).  For example, [`c2pa/c2pa_watermark_example.py`](https://github.com/adobe/trustmark/blob/main/c2pa/c2pa_watermark_example.py) shows how to reflect the identifier within the C2PA manifest using a _soft binding assertion_.

For more information, see the [FAQ](../FAQ.md#how-does-trustmark-align-with-provenance-standards-such-as-the-c2pa).

## Signpost watermark

TrustMark [coexists well with most other image watermarks](https://arxiv.org/abs/2501.17356) and so can be used as a _signpost_ to indicate the co-presence of another watermarking technology.  This can be helpful, since as an open technology, TrustMark can be used to indicate (signpost) which decoder to obtain and run on an image to decode a soft binding identifier for C2PA.

In this mode the encoding should be `Encoding.BCH_SUPER` and the payload contain an integer identifier that describes the co-present watermark.  The integer should be taken from the registry of C2PA-approved watermarks listed in this normative C2PA [softbinding-algorithms-list](https://github.com/c2pa-org/softbinding-algorithms-list) repository.
