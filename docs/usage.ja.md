# Krea 2 Identity Edit for Forge Neo 利用ガイド

このextensionは、Krea 2 Identity Edit LoRAに必要な2つの参照経路をForge Neoの
txt2imgへ追加する。

- appearance path：参照画像をclean VAE latent tokenとしてDiTへ渡す
- semantic path：編集指示と参照画像をQwen3-VLへ同時に渡す

元になったComfyUI実装は
[`lbouaraba/comfyui-krea2edit`](https://github.com/lbouaraba/comfyui-krea2edit)
である。

## 1. 必要なモデル

- Krea 2 RawまたはKrea 2 Turbo
- Krea 2用Qwen3-VL text encoder
- Krea 2用VAE
- `krea2_identity_edit_v1_2.safetensors`

Identity Edit LoRAはextensionへ同梱されない。Forgeの通常のLoRAフォルダーへ配置する。

## 2. 単一参照の基本操作

1. Forge NeoでKrea 2 RawまたはTurboを選択する。
2. txt2imgの **Krea 2 Identity Edit** accordionを開いて有効にする。
3. **Subject reference (required)**へ編集元画像をアップロードする。
4. 通常のpositive promptへ編集指示を書く。
5. positive prompt内でIdentity Edit LoRAを有効にする。
6. 2MP以下の解像度で生成する。

```text
<lora:krea2_identity_edit_v1_2:1>
```

開始設定の目安：

| Model | Steps | CFG | 用途 |
|---|---:|---:|---|
| Krea 2 Turbo | 8 | 1 | 通常の編集、短時間の試行 |
| Krea 2 Raw | 約20 | 約3 | 除去など、強いprompt追従が必要な編集 |

promptには完成画像の説明だけでなく、何を維持して何を変更するかを明示する。

```text
Keep the person's identity and facial features. Change the jacket to a dark
green coat and place the person in a softly lit train station.
<lora:krea2_identity_edit_v1_2:1>
```

## 3. 2参照編集

Forge UIではsubjectを必須、sceneを任意として入力する。2参照時は学習時の順序へ合わせる
ため、extension内部でscene→subjectの順に並べ替える。

| Forge UI | 役割 | RoPE frame | ComfyUI版の入力 |
|---|---|---:|---|
| Subject reference（必須） | 人物／主対象 | 単一時1、2参照時2 | 単一時main、2参照時`_b` |
| Scene reference（任意） | 構図／背景 | 1 | `source_image`, `source_latent` |

UIではSubject referenceを先に選ぶが、2参照時のDiTとQwen3-VLには
`[scene, subject]`の順で渡される。

```text
Place the person from the subject reference into the provided scene. Preserve
the person's face, hair, and clothing details.
<lora:krea2_identity_edit_v1_2:1>
```

Scene referenceを空にすると単一参照モードになる。2人を扱う場合も、別々の生成へ分ける
よりA/Bを同じsamplingへ同時に渡す方が、参照間の関係を指示しやすい。ただし顔の
分離は完全には保証されない。

## 4. UI設定

### Grounding resolution

Qwen3-VLへ入力する参照画像の長辺上限。既定値は`768`。

- 小さくする：編集指示へ追従しやすい
- 大きくする：人物のidentityや細部を拾いやすいが、VRAMと処理時間が増える

まず`768`を使い、顔の保持が不足するときだけ`1024`前後を試す。

### Subject reference boost

targetからsubject referenceへのattentionを強める。

- 単一／2参照ともSubject referenceへ適用
- `1.0`：無効
- `1.0`より大きい：参照の外見を強く保持
- `1.0`より小さい：参照から離れやすくする

`1.0`から始め、`1.5`、`2.0`のように少しずつ上げる。高すぎる値では編集が弱く
なったり、参照内容が出力へbleedしたりする場合がある。

### Scene reference boost

任意のScene referenceへ適用する。単一参照では効果がない。既定値`1.0`から、sceneの
構図や背景が失われる場合だけ上げる。

### Subject boost mask

Subject reference内でSubject reference boostを適用する領域を指定する。

- 白：boost対象
- 黒：boost対象外
- 判定しきい値：おおむね50%

単一／2参照とも必須のSubject referenceに対応する。これは出力画像の編集領域maskでは
なく、参照画像のどの情報を強く読むかを指定するattention maskである。
参照と同じcanvas／aspect ratioのグレースケール画像を推奨する。

### Grounding system prompt override

Qwen3-VLが参照画像の何へ注目するかを調整する上級者向け設定。空欄ではForgeの標準
image-grounding templateを保持する。

```text
Describe facial identity, hair, clothing, body pose, lighting, and spatial
relationships precisely.
```

通常は空欄のまま使用する。変更するとComfyUI版や既存seedとの一致度が変わる。

## 5. 追加LoRAとの併用

すべてのmodel LoRAは`[text | reference | target]`の同一DiT系列へ適用される。このため、
キャラクターLoRAやstyle LoRAはtargetだけでなくreferenceの解釈にも影響し得る。

追加LoRAのtext encoder weightが参照画像を変質させる場合は、TE strengthを`0`にする。

```text
<lora:krea2_identity_edit_v1_2:1>
<lora:additional_lora:te=0:unet=0.3>
```

`te=0`でもDiT側の影響は残る。追加LoRAの`unet`値は低い値から調整する。

## 6. ComfyUI版との比較

| 項目 | Forge Neo版 | ComfyUI-Krea2Edit |
|---|---|---|
| 操作方法 | txt2img内の1つのaccordion | Model PatchとGrounded Encodeノードを配線 |
| Identity Edit LoRA | promptの通常LoRA記法 | `LoraLoaderModelOnly`を推奨 |
| 単一参照 | Subject reference（必須） | mainの`source_image` / `source_latent` |
| 2参照 | Subject必須、Scene任意。内部でscene→subject | main=scene、`_b`=subject |
| RoPE順序 | target=0、A=1、B=2 | 同じ |
| Qwen画像順序 | A→Bを自動注入 | Grounded Encodeへscene→subject順に接続 |
| Negative grounding | positive/negativeへ自動的に同じ参照列を注入 | CFG>1では空promptのGrounded Encodeを別途配線 |
| `ref_boost` | Subject reference boost、UI上限16 | `ref_boost`、UI上限1000 |
| `ref_boost_a` | Scene reference boost | `ref_boost_a` |
| boost mask | グレースケール画像をアップロード | ComfyUI `MASK`入力 |
| reference geometry | v1.2 `fit`のみ | `fit`とlegacy `crop` |
| system prompt | 空欄でForge標準を維持、入力時のみoverride | 空欄でノード既定のtraining prompt |
| batch | `batch_size=1`、複数枚は`n_iter` | graphとmodel側の対応範囲で利用 |
| Hires. fix | 無効 | Forge固有機能のため該当なし |
| API | extension固有画像入力は未対応 | ComfyUI workflow/APIとして利用可能 |

Forge UIの並びは操作上重要なsubjectを先に表示する。一方、内部系列はComfyUI版と同じ
scene→subject順であり、UIの表示順をそのままtoken順には使用しない。

### 同等になる処理

両実装とも、基本的なsampling系列は次の形になる。

```text
[Qwen grounded text | scene(frame=1) | subject(frame=2) | target(frame=0)]
```

最後にtarget tokenだけを画像latentへ戻す。2参照の順序、fit geometry、boost値、prompt、
seed、samplerを揃えると比較しやすい。

### 完全一致しない主な理由

- ForgeとComfyUIでattention backend、dtype、乱数生成が異なる
- system promptの空欄時の扱いが異なる
- Forgeではpositive/negative groundingが自動化されている
- Forge版にはlegacy `crop`がない
- 追加LoRAのloaderとTE/DiT strength指定方法が異なる

したがって、同じseedでもpixel単位の一致は期待せず、identity保持、scene保持、編集追従
を比較する。

## 7. トラブルシューティング

### Sceneの人物がSubjectより強く出る

- SubjectとSceneの入力欄を取り違えていないか確認する
- Subject reference boostを少し上げる
- Scene reference boostを`1.0`へ戻す
- prompt内で「subject referenceの人物」と明記する

### 編集されず元画像に近すぎる

- Subject reference boostを`1.0`へ戻す、または下げる
- Grounding resolutionを下げる
- 強い除去編集ではRaw、CFG約3、約20 stepsを試す

### 顔や衣装が崩れる

- Grounding resolutionを上げる
- Subject reference boostを段階的に上げる
- 追加LoRAを弱め、必要なら`te=0`にする
- 出力を2MP以下にする

### boost使用時にVRAMが不足する

boostはsequence全体の加算attention biasを使用する。2参照・高解像度では消費量が
大きくなる。

- 出力解像度を下げる
- boostを`1.0`へ戻す
- 2参照が不要なら単一参照へ戻す
- まず単一参照で構図とpromptを確認する

### boost値を変えても結果が変わらない

bundled SageAttention 1.0.6は加算attention maskを無視するため、reference boostが効かない。
`webui-user.bat`でSageAttentionを無効化し、Forgeを再起動する。

```bat
set COMMANDLINE_ARGS=--uv --api --disable-sage
```

起動ログで`Using PyTorch Cross Attention`を確認する。Scene referenceがない場合、Scene
reference boostは仕様上効果がない。

## 8. 現在の制約

- txt2imgのみ
- v1.2 `fit` geometryのみ
- batch size 1
- Hires. fix非対応
- 最大2,000,000 output pixels
- extension固有reference入力のWeb API対応なし
- 実モデルのgolden image regressionは未整備
- SageAttention 1.0.6ではreference boost無効。`--disable-sage`が必要
