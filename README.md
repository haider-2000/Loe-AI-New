# Leo — EduData AI Bot

بوت Telegram تعليمي يعمل داخل مجموعة Telegram واحدة فقط، ويستخدم Google Gemini لفهم الأسئلة النصية والصور والرسائل الصوتية والرد باللهجة العراقية عند استخدامها. المشروع يستخدم Python 3.11+ وSQLite فقط، ولا يحتوي على Flask أو FastAPI أو لوحة تحكم أو قاعدة بيانات خارجية.

## هوية البوت

- **الاسم:** Leo
- **المعرّف:** [@Leo_Al_Best_Assistant_bot](https://t.me/Leo_Al_Best_Assistant_bot)
- **الوصف:** أنا Leo، بوت بنيت من قبل . Haider Aqeel Falih رابط حسابه @Haider_Aqeel. بقدر أساعدك في أي حاجة تحتاجها — أسئلة، معلومات، أي شي.
- **العبارة الترحيبية:** كيف أساعدك اليوم؟

## الوظائف

يدعم `/start` و`/help` و`/privacy` و`/status` و`/cancel`، إضافة إلى تنظيم الحصص حتى ٥٠ طالبًا للحصة. الأمر `/lessons` يعرض الجدول والمقاعد المتبقية. المدير يستخدم `/newlesson المادة | 2026-10-01 | 10:00 | 50` لإنشاء حصة، ثم `/addstudent رقم_الحصة | رمز_الطالب` للتسجيل و`/roster رقم_الحصة` لعرض القائمة. الأوامر `/stats` و`/pending` و`/export` متاحة فقط للحساب الذي يطابق `ADMIN_ID`. في المجموعات لا يرد البوت على كل رسالة؛ بل يرد فقط عند ذكر username مالته، أو عند الرد المباشر على رسالة منه. **يعمل الآن في أي مجموعة أو مجموعة خارقة يضاف إليها البوت**، بلا قائمة مجموعات مسموح بها.

النصوص والصور التعليمية تُسجّل بحالة `review`. الصوت يُنزّل إلى الذاكرة فقط، ويُرسل إلى Gemini للفهم، ثم تُحفظ الترجمة النصية المحتملة فقط. لا توجد خانة `audio` في قاعدة البيانات ولا مجلد `data/raw/audio/` ولا أمر `/contribute` ولا رسالة طلب موافقة.

## الخصوصية

يُمنع إدخال Telegram IDs وusernames وأسماء الطلاب وأرقام الهواتف والعناوين والملفات الصوتية في dataset. تُفحص النصوص بحثًا عن مؤشرات شخصية واضحة، كما يُجرى فحص بصري للصور عبر Gemini بحثًا عن الأسماء أو الهواتف أو الهويات أو العناوين أو الوجوه. عند الاشتباه، تكون `privacy_flag=1` ويُستبعد السجل من التصدير. تبقى الصورة داخليًا للمراجعة ولا تُصدّر تلقائيًا.

## التثبيت على Linux/macOS

```bash
cd edu_bot
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

افتح `.env` وضع:

```dotenv
TELEGRAM_BOT_TOKEN=توكن BotFather
GEMINI_API_KEY=مفتاح Google AI Studio
ADMIN_ID=رقم Telegram الرقمي لحساب المدير
GEMINI_MODEL=gemini-3.5-flash-lite
```

ثم شغّل:

```bash
python bot.py
```

## التثبيت على Windows PowerShell

```powershell
cd edu_bot
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python bot.py
```

إذا منع PowerShell تفعيل البيئة، نفّذ مرة واحدة في نافذة PowerShell مناسبة: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## المراجعة والتصدير

تُحفظ السجلات في SQLite بالحالة `review`. بعد مراجعة بشرية، يعتمد المدير السجل بتحديث SQLite مثلًا:

```sql
UPDATE contributions SET status = 'approved' WHERE id = 123 AND privacy_flag = 0;
```

ثم يرسل المدير `/export` للبوت. ينتج JSONL في `data/exports/` ولا يُصدّر إلا `approved` و`privacy_flag=0`. لا يحتوي التصدير على أي معرفات Telegram أو أسماء مستخدمين. بيانات الحصص تستخدم رمزًا يحدده المدير بدل أسماء الطلاب أو معرفات Telegram، حفاظًا على الخصوصية.

## الاختبار المحلي

للفحص دون مفاتيح أو اتصال Telegram:

```bash
python -m py_compile bot.py config.py database.py dataset.py gemini_client.py
python - <<'PY'
import asyncio
from database import init_db, add_contribution, count_by_status, export_jsonl

async def main():
    db = 'data/test.db'
    await init_db(db)
    await add_contribution(db, contribution_type='text', file_path=None,
        text_content='شلون أحل هاي المسألة؟', transcription=None,
        model_answer='نحلها خطوة خطوة.', language='ar', dialect='iraqi', privacy_flag=False,
        status='approved')
    print(await count_by_status(db))
    print(await export_jsonl(db, 'data/exports/test.jsonl'))

asyncio.run(main())
PY
```

احذف `data/test.db` و`data/exports/test.jsonl` بعد الاختبار. لا تشغّل البوت قبل ضبط المتغيرات المطلوبة، وبالأخص `ADMIN_ID`.

## ملاحظات تشغيلية

يحتاج البوت إلى صلاحية قراءة الرسائل المناسبة في المجموعة وفق إعدادات Telegram Privacy Mode. للحصول على رسائل المجموعة الموجّهة إليه فقط، يمكن إبقاء Privacy Mode مفعّلًا؛ وإذا احتجت معالجة أوسع فاضبط إعدادات BotFather بعناية. لا تُضمّن ملف `.env` في Git.

## النسخ الاحتياطي: بياناتك على لابتوبك

قرص Render مؤقت ويمسح كل شيء بعد أي إعادة نشر أو إعادة تشغيل. لذلك:

1. أرسل `/backup` **على الخاص** (محادثة مباشرة مع البوت). يبني البوت ملف zip فيه قاعدة البيانات كاملة والصور و`manifest.json` بأعداد العناصر، ويرسله لك كملف — نزّله على لابتوبك.
2. _archive_ محلي: فك الضغط وضع `edu_bot.db` و`images/` داخل مجلد `seed/` في المشروع، ثم ارفعه إلى Git. عند الإقلاع على قرص فارغ يقرأ البوت `seed/` تلقائيًا فيسترجع البيانات بدل ما يبدأ من صفر.

`/backup` غير مربوط بأي قائمة أوامر (`set_my_commands` غير مستخدم)، فلا يظهر في قائمة `/` لأي طالب. وهو صامت بالمجموعة تمامًا: لا يردّ ولا يهمس، لأن أي رسالة مثل «هذا الأمر للخاص» كانت بحد ذاتها تكشف وجوده. أي شخص غير `ADMIN_ID` يُتجاهل بصمت حتى على الخاص.

## من يقدر يستخدم البوت

| المكان | مين يمر |
|---|---|
| أي مجموعة أو مجموعة خارقة | أي طالب، لكن فقط إذا ذُكر البوت بالرسالة أو رُدّ عليه |
| الخاص | **أنت فقط** (`ADMIN_ID`)، بكل الأوامر |
| أي خاصة ثانية أو مجموعة ثانية | يُتجاهل بصمت |

وجودك بالمجموعة مثل بقية الطلاب: لا تمييز، فقط تذكر `@Leo_AI_Best_Assistant_bot` عشان يرد عليك.

لا يستبدل `seed/` أبدًا قاعدة بيانات موجودة، فلن يفقدك بيانات أحدث من النسخة المحفوظة.

## وين تعيش البيانات

الكود نفسه يشتغل بقاعدتين، والفرق بمتغير واحد:

| المتغير | وين تصير البيانات | ليش |
|---|---|---|
| `DATABASE_PATH=data/edu_bot.db` | ملف SQLite على جهازك | للتطوير والاستخدام اليومي |
| `DB_URL=libsql://….turso.io` | قاعدة بيانات سحابية مجانية | لو على Render، لأن قرص الحاوية ينمسح |

لما `DB_URL` يبدأ بـ `libsql://` (أو `https://`)، `database.py` يشتغل عبر `libsql-client` بدل `aiosqlite`. باقي الكود ما يتغير.

**الصور مخزّنة داخل القاعدة** (جدول `images`، عمود BLOB)، لأن الصف الي يشير لصورة تنطفي مع القرص. عند الإقلاع، `materialize_images()` يرجّع ملفات الصور من القاعدة إلى القرص.

## وينام Render ويصحى (الخطة المجانية)

خدمة Render المجانية:

- تتوقف بعد **١٥ دقيقة** بدون زيارات، وأي توقف **يمسح قرص الحاوية بالكامل**.
- ما تقبل قرص دائم على Free.

الحل المطبّق هنا:

1. **البيانات برّا الحاوية** (`DB_URL`) — تنجو من أي مسح.
2. **`/health`** — البوت يفتح منفذ ويجاوب `ok`، لأن بوت long polling ما يستقبل أي طلب داخلي، و Render يحتاج منفذ للراوتنق.
3. **self-ping** — كل ٤ دقايق البوت يطلب `/health` عند نفسه، فما يگدر يگرب إنه خامل.
4. **backup تلقائي** كل `AUTO_BACKUP_MINUTES` دقايق، بس لمّا يجي محتوى جديد، ويوصلك على الخاص.

## أول نشر: انقل بياناتك

الـ ٢٤ عنصر الموجودة على لابتوبك ما تنتقل وحدها. قبل أول deploy:

```bash
# 1) ضع المفاتيح بـ .env
#    DB_URL=libsql://your-db-your-org.turso.io
#    DB_AUTH_TOKEN=eyJhbGciOi...
# 2) انقل البيانات
python migrate_to_remote.py
```

السكربت يرفض الكتابة على قاعدة فيها بيانات، إلا إذا مرّرت `--force`. و `/backup` يبني نسخة SQLite حقيقية حتى لمّا تكون البيانات سحابية، فالملف الي تحفظه على لابتوبك يفتح بأي برنامج.

## النشر على Render

الملف `render.yaml` جاهز (Blueprint) ومو صفّر خدمة `web` على `plan: free`:

1. ارفع المشروع على GitHub.
2. Render Dashboard ← **New** ← **Blueprint** ← اختر الريبو (وحط allow للريبو الخاص).
3. Render يقرا `render.yaml` ويسأل عن Secrets. دخّل:
   - `TELEGRAM_BOT_TOKEN`
   - `GEMINI_API_KEY`
   - `ADMIN_ID`
   - `DB_URL` ← رابط Turso
   - `DB_AUTH_TOKEN` ← توكن Turso
4. اضغط **Apply**. راقب اللوق: `Health endpoint listening on port …` ثم `Database initialized` ثم `Application started`.
5. بعد أول deploy، جرّب البوت بالمجموعة، وبعدين تأكد من الإحصائيات.

ملاحظات:

- `plan: free` يعطيك ٧٥٠ ساعة بالشهر، وهي تكفي خدمة وحدة تشتغل ٢٤ ساعة.
- Render يوقف أي خدمة Free بعد ١٥ دقيقة بدون **طلبات واردة**، والطلب اللي يطلع من نفس الـ container ما ينحسب. لذلك `bot.py` يندك على `RENDER_EXTERNAL_URL` (يعطيه Render تلقائيًا) كل ٤ دقائق، مو على `127.0.0.1`. لو شفت البوت يرد وبعدين يطفي، هذا غالبًا السبب.
- في نفس الوقت، خلّي مراقب خارجي (مثل UptimeRobot) يندك على رابط الخدمة `/health` كل ٥ دقائق: هذا يوقظ الخدمة لو الـ container طفا، والبوت ما يقدر يوقظ نفسه.
- الـ Free filesystem مؤقت: أي ملف ينكتب على القرص يروح مع كل spin-down أو redeploy. البيانات محفوظة ب Turso مو على القرص.
- لو مرّ وقت والإحصائيات صفر، راجع اللوق: إما البوت ليس عضواً في المجموعة، أو `ADMIN_ID` غلط، أو `DB_URL` فاضي.
- احتفظ دائمًا بنسخة على لابتوبك: النسخ التلقائي كل ١٥ دقيقة موجود كشبكة أمان، بس لا تعتمد عليه وحده.
