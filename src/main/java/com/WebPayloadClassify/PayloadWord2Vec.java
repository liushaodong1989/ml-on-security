package com.WebPayloadClassify;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.net.URLDecoder;
import java.util.Collection;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


/**
 * Created by shad.liu on 17/6/30.
 */
public class PayloadWord2Vec {
    private static Logger log = LoggerFactory.getLogger(PayloadWord2Vec.class);


    //训练模型
    public static Word2Vec trainWord2VecModel()throws Exception{
        //语料库使用的文件
        String tmp_dir = PayloadDataFetcher.PAYLOAD_ROOT;
        int layer_size = PayloadDataFetcher.word2vecColumns;

        log.info("Load & Vectorize Sentences....");
        SentenceIterator iter = new BasicLineIterator(tmp_dir+"all.txt_decode.txt");
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new preStringHandler());

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
            .minWordFrequency(3)
            .iterations(1)
            .layerSize(layer_size)
            .seed(42)
            .windowSize(8)
            .iterate(iter)
            .tokenizerFactory(t)
            .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();
        log.info("Writing word vectors to text file....");
        WordVectorSerializer.writeFullModel(vec, tmp_dir + "payladWord2Vec.model");

        return vec;
    }

    private static String charNormalize(String test){
        //去char chr处理
        String result = test;

        while(true){
            String regx = "CH[A]*R\\(\\d+\\)";

            Pattern pat = Pattern.compile(regx, Pattern.CASE_INSENSITIVE);
            Matcher matcher = pat.matcher(result);

            StringBuffer sb = new StringBuffer();

            boolean res = matcher.find();
            if(!res){
                break;
            }

            while(res){
                String tmp = matcher.group();

                //继续使用正则将数组取出来
                String regx1 = "\\d+";
                Pattern pat1 = Pattern.compile(regx1);
                Matcher matcher1 = pat1.matcher(tmp);

                boolean result1 = matcher1.find();
                if(result1){
                    String numStr = matcher1.group();
                    //System.out.println(numStr);

                    char c = (char)Integer.parseInt(numStr);
                    matcher.appendReplacement(sb, String.valueOf(c));

                }
                res = matcher.find();
            }
            matcher.appendTail(sb);
            result = sb.toString();
        }
        return result;
    }

    public static String unicodeToUtf8(String theString) {
        char aChar;
        int len = theString.length();
        StringBuffer outBuffer = new StringBuffer(len);
        for (int x = 0; x < len;) {
            aChar = theString.charAt(x++);
            if (aChar == '\\') {
                aChar = theString.charAt(x++);
                if (aChar == 'u') {
                    // Read the xxxx
                    int value = 0;
                    for (int i = 0; i < 4; i++) {
                        aChar = theString.charAt(x++);
                        switch (aChar) {
                            case '0':
                            case '1':
                            case '2':
                            case '3':
                            case '4':
                            case '5':
                            case '6':
                            case '7':
                            case '8':
                            case '9':
                                value = (value << 4) + aChar - '0';
                                break;
                            case 'a':
                            case 'b':
                            case 'c':
                            case 'd':
                            case 'e':
                            case 'f':
                                value = (value << 4) + 10 + aChar - 'a';
                                break;
                            case 'A':
                            case 'B':
                            case 'C':
                            case 'D':
                            case 'E':
                            case 'F':
                                value = (value << 4) + 10 + aChar - 'A';
                                break;
                            default:
                                throw new IllegalArgumentException(
                                    "Malformed   \\uxxxx   encoding.");
                        }
                    }
                    outBuffer.append((char) value);
                } else {
                    if (aChar == 't')
                        aChar = '\t';
                    else if (aChar == 'r')
                        aChar = '\r';
                    else if (aChar == 'n')
                        aChar = '\n';
                    else if (aChar == 'f')
                        aChar = '\f';
                    outBuffer.append(aChar);
                }
            } else
                outBuffer.append(aChar);
        }
        return outBuffer.toString();
    }

    /*
         * 字符转换为字节
         */
    private static byte charToByte(char c) {
        return (byte) "0123456789ABCDEF".indexOf(c);
    }

    /*
     * 16进制字符串转字节数组 0x2d这种串
     */
    public static byte[] hexString2Bytes(String hex) {

        if ((hex == null) || (hex.equals(""))){
            return null;
        }
        else if (hex.length()%4 != 0){
            return null;
        }
        else{
            hex = hex.toUpperCase();
            int len = hex.length()/4;
            byte[] b = new byte[len];
            char[] hc = hex.toCharArray();
            for (int i=0; i<len; i++){
                int p=4*i;
                b[i] = (byte) (charToByte(hc[p+2]) << 4 | charToByte(hc[p+3]));
            }
            return b;
        }

    }
    /*
    * 字节数组转字符串
    */
    public static String bytes2String(byte[] b) throws Exception {
        String r = new String (b,"UTF-8");
        return r;
    }

    public static String hexStringToString(String s){
        try {

            return bytes2String(hexString2Bytes(s));
        }catch (Exception e){
            return s;
        }
    }

    public static String hexStringsToStrings(String result){
        //用正则的方法来处理
        //url 编码处理
        String regx = "(0[x|X][\\da-fA-F][\\da-fA-F])";
        Pattern pat = Pattern.compile(regx, Pattern.CASE_INSENSITIVE);
        Matcher matcher = pat.matcher(result);
        StringBuffer sb = new StringBuffer();

        boolean res = matcher.find();
        if (!res) {
            return result;
        }

        while (res) {
            String tmp = matcher.group();

            try {
                String replace = hexStringToString(tmp);
                matcher.appendReplacement(sb, replace);
            } catch (Exception e) {
                matcher.appendReplacement(sb, " ");
            }

            res = matcher.find();
        }
        matcher.appendTail(sb);

        return sb.toString();
    }

    public static String hexStringsToStrings2(String result){
        //用正则的方法来处理
        //url 编码处理
        String regx = "(/[x|X][\\da-fA-F][\\da-fA-F])";
        Pattern pat = Pattern.compile(regx, Pattern.CASE_INSENSITIVE);
        Matcher matcher = pat.matcher(result);
        StringBuffer sb = new StringBuffer();

        boolean res = matcher.find();
        if (!res) {
            return result;
        }

        while (res) {
            String tmp = matcher.group();

            try {
                String replace = hexStringToString(tmp);
                matcher.appendReplacement(sb, replace);
            } catch (Exception e) {
                matcher.appendReplacement(sb, " ");
            }

            res = matcher.find();
        }
        matcher.appendTail(sb);

        return sb.toString();
    }
    private static String symbolAddBlank(String test){
        //将单词分隔开（麻烦）
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i< test.length(); i++){
            char c = test.charAt(i);

            boolean is_letter = Character.isLetter(c) || Character.isDigit(c) || c == ' ';


            //非字母
            if(!is_letter){
                sb.append(' ');
                sb.append(c);
                sb.append(' ');
            }else{
                sb.append(c);
            }
        }

        return sb.toString();
    }

    /*
   * base64解码
    */
    public static String tryBase64Decode(String payload){
        //用正则的方式处理
        String regx = "([A-Za-z0-9+\\/]{4}){8,}([A-Za-z0-9+\\/]{4}|[A-Za-z0-9+\\/]{3}=|[A-Za-z0-9+\\/]{2}==)"; //先用正则简单的判断是不是可能的base64编码
        Pattern pat = Pattern.compile(regx, Pattern.CASE_INSENSITIVE);
        Matcher matcher = pat.matcher(payload);
        StringBuffer sb = new StringBuffer();

        boolean res = matcher.find();
        if (!res) {
            return payload;
        }

        while (res) {
            String tmp = matcher.group();

            try {
                //base64编码的 肯定同时存在大写和小写的，否则不是base64编码，是其他客户端产生的随机数
                String tmp_upper = tmp.toUpperCase();
                String tmp_low = tmp.toLowerCase();
                String replace = tmp;
                //进一步判断
                if(!(tmp_low.equals(tmp) || tmp_upper.equals(tmp))){
                    replace =  new String(org.apache.commons.codec.binary.Base64.decodeBase64(tmp), "UTF-8");

                }

                //有$ 时这个操作不行
                matcher.appendReplacement(sb, Matcher.quoteReplacement(replace));

            } catch (Exception e) {
                e.printStackTrace();
                matcher.appendReplacement(sb, " " + tmp + "");
            }

            res = matcher.find();
        }

        matcher.appendTail(sb);
        return sb.toString();
    }

    public static String _URLDecode(String test) {
        //用正则方法来处理，攻击者可能会嵌套urlencode
        String result = test;
        int index = 0;
        while (index < 10) {
            //%25要处理
            if(result.contains("%25")){
                result = result.replace("%25", "%");
            }

            //%u unicode编码处理
            if(result.contains("%u")  ){
                result = result.replace("%u", "\\u");
            }

            if(result.contains("/u")){
                result = result.replace("/u", "\\u");
            }

            try {
                if (result.contains("\\u")) {
                    result = unicodeToUtf8(result);
                }
            }catch(Exception e){
                // nothing
            }

            //16进制解码
            if(result.contains("0x") || result.contains("0X")){
                result = hexStringsToStrings(result);
            }

            //16进制 /x61 这种形式的 需要用正则来处理
            if(result.contains("/x") || result.contains("/X") || result.contains("\\x") || result.contains("\\X")){
                result = hexStringsToStrings2(result);
            }

            //url 编码处理
            String regx = "(%[\\da-zA-Z][\\da-zA-Z]){1,}";
            Pattern pat = Pattern.compile(regx, Pattern.CASE_INSENSITIVE);
            Matcher matcher = pat.matcher(result);
            StringBuffer sb = new StringBuffer();

            char c = 0xFFFD;
            String tmpRegx = String.valueOf(c);

            boolean res = matcher.find();
            if (!res) {
                break;
            }

            while (res) {
                String tmp = matcher.group();
                //换行的特殊处理
                if(tmp.toLowerCase().compareTo("%1f") == -1){
                    tmp = "%20";
                }

                try {
                    String replace = URLDecoder.decode(tmp, "utf-8");
                    //对于转换错误的直接用空格代替
                    replace = replace.replaceAll(tmpRegx, " ");
                    matcher.appendReplacement(sb, Matcher.quoteReplacement(replace));

                } catch (Exception e) {
                    matcher.appendReplacement(sb, " ");
                }

                res = matcher.find();
            }
            matcher.appendTail(sb);

            result = sb.toString();
            index++;
        }

        result = tryBase64Decode(result);

        result = result.replaceAll("\\+", " ");
        return result.toLowerCase();
    }


    public static String payloadNormalize(String test){
        String result = charNormalize(test);
        return symbolAddBlank(result);
    }

    /*
    *生成all.txt_decode.txt文件
     */
    public static void generateDecodePayload() throws Exception{
        String tmp_dir = PayloadDataFetcher.PAYLOAD_ROOT;

        //读取4个文件生成decode文件
        File result_file = new File(tmp_dir + "all.txt_decode.txt");
        FileWriter fw = new FileWriter(result_file);

        String[] paths = {"good-payloads.txt", "sqli-payloads.txt","xss-payloads.txt","rmea-payloads.txt"};
        for(int i =0; i < paths.length; i++){
             File file = new File(tmp_dir + paths[i]);
             BufferedReader  br = new BufferedReader(new FileReader(file));

             String line = br.readLine();
             //System.out.println(line);
             while(line != null){
                 line = _URLDecode(line);
                 line = payloadNormalize(line);
                 fw.write(line + "\n");
                 line = br.readLine();
             }
             br.close();
        }
        fw.close();
    }


    public static String[] normalize(Integer nums, String line){
        String payload = _URLDecode(line);
        payload = payloadNormalize(payload);

        return payload.split(" ");
    }


    public static void main(String[] args) throws Exception {

        String tmp_root = PayloadDataFetcher.PAYLOAD_ROOT;

        File f0 = new File(tmp_root + "all.txt_decode.txt");
        if(!f0.exists()){
            System.out.println("should generate all.txt_decode.txt");
            generateDecodePayload();
        }


        Word2Vec word2Vec = null;
        //判断语料库是否存在
        File f = new File(tmp_root + "payladWord2Vec.model");
        if(f.exists()) {
            word2Vec = WordVectorSerializer.loadFullModel(tmp_root + "payladWord2Vec.model");
        }else{
            word2Vec = trainWord2VecModel();
        }

        Collection<String> lst1 = word2Vec.wordsNearest("select", 20);
        System.out.println("20 Words closest to 'select': " + lst1);

        Collection<String> lst2 = word2Vec.wordsNearest("script", 20);
        System.out.println("20 Words closest to 'script': " + lst2);
    }
}
