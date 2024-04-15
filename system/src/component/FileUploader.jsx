import React, { useState } from 'react';

const FileUploader = ({ selectedModel }) => {
  const [fileContent, setFileContent] = useState({ content: '' });
  const [apiResponse, setApiResponse] = useState(null);
  const [selectedOption, setSelectedOption] = useState("default");
  const [error, setError] = useState(null);
  const [showApiResponse, setShowApiResponse] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleOptionChange = (event) => {
    setSelectedOption(event.target.value);
    updateTextAreaContent(event.target.value);
    setShowApiResponse(false);
  };

  const updateTextAreaContent = (selectedOption) => {
    let exampleText = "";
    switch (selectedOption) {
      case "option1":
        exampleText = "Chúng tôi phát hiện N-nitrosodiethylamine (NDEA) trong số thuốc tim valsartan do Công ty Dược phẩm Chiết Giang Huahai (ZHP) sản xuất bằng quy trình cũ trước khi thay đổi vào năm 2012. Hiện dữ liệu về mức độ NDEA rất hạn chế, Fox News dẫn thông tin từ Cơ quan Dược phẩm châu Âu (EMA). Theo Tổ chức Y tế Thế giới (WHO), NDEA bị xếp vào danh sách chất có thể gây ung thư cho người. Chất này thường xuất hiện trong khói thuốc lá. Trước đó, từ tháng 7, valsartan của ZHP đã bị phát hiện chứa N-nitrosodimethylamine (NDMA) và bị thu hồi trên khắp thế giới. Nghiên cứu trên động vật chỉ ra NDMA gây ung thư gan, thận và đường hô hấp. Dùng để điều trị huyết áp cao và phòng ngừa suy tim, valsartan ban đầu được phát triển bởi công ty Novartis (Thụy Sĩ) dưới tên thương mại Diovan. Hết hạn bảo hộ độc quyền, nhiều hãng dược phẩm khác tham gia điều chế valsartan, trong đó ZHP là nhà sản xuất lớn ở Trung Quốc và cung cấp cho hầu hết thị trường thế giới. EMA cho biết sẽ thu thập thêm dữ liệu về nồng độ NDEA để đánh giá mức độ rủi ro. Cơ quan này cũng xác nhận nguy cơ mắc ung thư từ NDMA trong valsartan là thấp. EMA đồng thời khuyến cáo bệnh nhân đang dùng thuốc tuyệt đối không bỏ đột ngột trừ khi đã tham khảo ý kiến bác sĩ. Cục Quản lý Dược Việt Nam từ đầu năm đến nay cũng nhiều lần thông báo thu hồi thuốc trị tim mạch huyết áp có chứa thành phần valsartan nguồn gốc từ ZHP.\n\n(We have detected N-nitrosodiethylamine (NDEA) in some valsartan heart medicines manufactured by Zhejiang Huahai Pharmaceutical (ZHP) using the old process before the change in 2012. Currently, data on the level of NDEA is very limited, Fox News cited information from the European Medicines Agency (EMA). According to the World Health Organization (WHO), NDEA is classified as a substance that can cause cancer in humans. This substance is often found in cigarette smoke. Previously, since July, ZHP's valsartan has been found to contain N-nitrosodimethylamine (NDMA) and has been recalled worldwide. Animal studies have shown that NDMA can cause cancer of the liver, kidneys, and respiratory tract. Used to treat high blood pressure and prevent heart failure, valsartan was originally developed by Novartis (Switzerland) under the trade name Diovan. After the expiration of patent protection, many other pharmaceutical companies joined in producing valsartan, among which ZHP is a major manufacturer in China and supplies to most of the world market. EMA stated that it would gather more data on the concentration of NDEA to assess the level of risk. The agency also confirmed that the risk of cancer from NDMA in valsartan is low. EMA also advises patients taking the medicine not to stop abruptly unless they have consulted their doctor. The Vietnam Drug Administration has also announced the recall of blood pressure and heart medicine containing valsartan from ZHP since the beginning of the year.)";
        break;
      case "option2":
        exampleText = "Hôm nay là ngày khó khăn nhất đời tôi và sẽ là ngày tôi chẳng bao giờ quên được, Stephanie Ray viết. Bên dưới những dòng chia sẻ ấy, thiếu nữ 15 tuổi đăng tấm ảnh cuối cùng chụp chung với bạn trai Blake Ward. Cách đó vài hôm, ngày 31/7, Blake bị nước cuốn khi đang bơi ở biển Tywyn, Cambrian News đưa tin. Khi được đưa đến bệnh viện, thiếu niên 16 tuổi đã bị tổn thương não quá nặng. Các bác sĩ nhận định Blake không có cơ hội nào hồi phục. Không muốn con chịu đựng đau đớn, ngày 4/8, gia đình Blake quyết định tắt thiết bị hỗ trợ sự sống để cậu bé ra đi thanh thản. Thường xuyên túc trực từ lúc bạn trai gặp nạn, Stephanie dù đau đớn nhưng cũng hiểu rằng đó là cách tốt nhất. Sau khi Blake mất, Stephanie vẫn lưu giữ mọi hình ảnh cặp đôi chụp cùng nhau. Trên trang cá nhân, Cô bé tuổi teen còn thực hiện một video ghi lại những khoảnh khắc hạnh phúc họ đã có với nhau. Blake là người vô cùng đặc biệt với tôi và mối liên kết giữa hai đứa sẽ chẳng bao giờ mất đi, Stephanie nghẹn ngào. Blake, anh sẽ luôn giữ một vị trí đặc biệt trong trái tim em. Em sẽ mãi mãi yêu anh.\n\n(Today is the hardest day of my life and will be a day I will never forget, Stephanie Ray wrote. Beneath those words, the 15-year-old girl posted the last photo taken with her boyfriend, Blake Ward. Just a few days ago, on July 31st, Blake was swept away by the water while swimming in the Tywyn sea, as reported by Cambrian News. When taken to the hospital, the 16-year-old boy suffered severe brain damage. Doctors determined that Blake had no chance of recovery. Not wanting her son to suffer, on August 4th, Blake's family decided to turn off life support so he could peacefully pass away. Stephanie, who had been vigilantly by her boyfriend's side since the accident, though in pain, understood that it was the best decision. Even after Blake's passing, Stephanie still cherishes every image of the couple taken together. On her personal page, the teenage girl also made a video capturing the happy moments they shared. 'Blake is incredibly special to me, and the bond between us will never be lost,' Stephanie choked up. 'Blake, you will always hold a special place in my heart. I will love you forever.)";
        break;
      case "option3":
        exampleText = "Xin chào, tôi là Chris Pratt. Hôm nay là ngày thứ ba tôi áp dụng chế độ ăn Daniel Fast. Hãy xem thử chế độ ăn kiêng 21 ngày này nhé, nam diễn viên Chris Pratt, ngôi sao Vệ binh Dải Ngân hà, chia sẻ trên mạng xã hội. Theo Men's Health, chế độ ăn Daniel Fast lấy cảm hứng từ Thánh Daniel, người được Kinh Cựu Ước mô tả là chỉ ăn rau và uống nước suốt 21 ngày đêm. Chế độ ăn này yêu cầu người thực hiện kiêng khem cực kỳ nghiêm ngặt, chỉ được ăn các loại hạt, rau củ quả và uống nước trong 21 ngày liên tục. Các sản phẩm từ động vật, sữa hoàn toàn bị cấm. Việc Chris Pratt áp dụng Daniel Fast khiến nhiều người chú ý hơn tới chế độ dinh dưỡng này đồng thời học theo. Tuy nhiên, các chuyên gia cảnh báo chế độ ăn này không hề tốt cho sức khỏe. Bà Liz Weinandy, bác sĩ dinh dưỡng tại Trung tâm y tế Wexner thuộc Đại học bang Ohio (Mỹ) cho biết chế độ ăn Daniel Fast tiềm ẩn vô số rủi ro, đặc biệt với những người đã từng gặp vấn đề sức khỏe. Chế độ ăn này thiếu tất cả các loại chất dinh dưỡng cần thiết bao gồm protein, chất béo và có thể dẫn tới thiếu dinh dưỡng trầm trọng, hạ natri máu, thậm chí tử vong, nữ bác sĩ lý giải. Theo bác sĩ Weinandy, nhịn ăn gián đoạn trong khoảng từ 12 đến 14 tiếng có thể đem lại lợi ích cho sức khỏe nhưng nếu kéo dài đến 21 ngày sẽ rất nguy hiểm. Tốt nhất, bạn hãy ăn uống điều độ và tránh xa các chế độ ăn kiêng quá khắc nghiệt.\n\n(Hello, I'm Chris Pratt. Today is my third day on the Daniel Fast diet. Check out this 21-day diet, 'said actor Chris Pratt, star of Guardians of the Galaxy, sharing on social media. According to Men's Health, the Daniel Fast diet is inspired by the Biblical figure Daniel, who ate only vegetables and drank water for 21 days and nights. This diet requires strict abstinence, allowing only consumption of grains, fruits, vegetables, and water for 21 consecutive days. Animal products and dairy are entirely prohibited. Chris Pratt's adoption of the Daniel Fast has drawn attention to this diet and inspired others to follow suit. However, experts warn that this diet is not healthy. Liz Weinandy, a nutritionist at the Wexner Medical Center at Ohio State University, notes that the Daniel Fast diet carries numerous risks, especially for those with pre-existing health issues. This diet lacks essential nutrients such as protein and fats and can lead to severe malnutrition, low blood sodium, and even death, explained Weinandy. While intermittent fasting for 12 to 14 hours may have health benefits, extending it to 21 days can be very dangerous. It's best to eat a balanced diet and stay away from overly strict diets.)";
        break;
      default:
        exampleText = "";
        break;
    }

    setFileContent({ content: exampleText });
  };

  const handleSendData = async () => {
    try {
      setShowApiResponse(false);
      setLoading(true); // Bắt đầu loading

      // Kiểm tra nếu nội dung của file là rỗng
      if (fileContent.content.trim() === '') {
        setError("Please enter some text before generating.");
        setLoading(false); // Dừng loading
        return; // Dừng việc gửi dữ liệu nếu textarea trống
      }

      let selectedId = 0; // Default id if no option is selected
      if (selectedOption !== "default") {
        // Get the corresponding id for the selected option
        switch (selectedOption) {
          case "option1":
            selectedId = 1;
            break;
          case "option2":
            selectedId = 2;
            break;
          case "option3":
            selectedId = 3;
            break;
          default:
            selectedId = 0;
            break;
        }
      }

      // Prepare data to send
      const dataToSend = {
        name: selectedModel,
        id: selectedId,
        data: fileContent.content
      };

      // Send data to API
      const response = await fetch('https://8b11-113-160-133-144.ngrok-free.app/gen', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(dataToSend),
      });

      const result = await response.json();
      console.log('API response:', result);

      setApiResponse(result.lowercase_data);
      setError(null);
      setShowApiResponse(true);
      setLoading(false); // Dừng loading khi nhận được kết quả từ API
    } catch (error) {
      console.error('Error sending data to API:', error);
      setError("Error sending data to API. Please try again.");
      setLoading(false); // Dừng loading nếu có lỗi
      setShowApiResponse(false);
    }
  };
  

  return (
    <div className="my-4">
      <div className="relative flex md:auto">
        <label
          htmlFor="exampleFormControlTextarea1"
          className="absolute top-2 right-2 text-neutral-500"
        >
        </label>

        <textarea
          value={fileContent.content}
          onChange={(e) => setFileContent({ content: e.target.value })}
          placeholder="Your message"
          id="exampleFormControlTextarea1"
          rows="10"
          readOnly={false}
          className="peer block min-h-[auto] w-full rounded border border-black bg-transparent px-3 py-[0.32rem] leading-[1.6] outline-none transition-all duration-200 ease-linear focus:placeholder:opacity-100 data-[te-input-state-active]:placeholder:opacity-100 motion-reduce:transition-none dark:text-neutral-200 dark:placeholder:text-neutral-200 [&:not([data-te-input-placeholder-active])]:placeholder:opacity-0"
        ></textarea>
      </div>
      <div className="relative flex md:auto">
        <select
          value={selectedOption}
          onChange={handleOptionChange}
          className="bg-white px-2 py-1 rounded-md absolute top-2 right-2"
        >
          <option value="default">Choose an example</option>
          <option value="option1">Option 1</option>
          <option value="option2">Option 2</option>
          <option value="option3">Option 3</option>
        </select>
      </div>

      {error && <div className="text-red-500 mt-2">{error}</div>}

      <div className="mt-4 flex pt-10 items-center justify-center mx-auto col-start-1 row-start-3 self-center sm:mt-0 sm:col-start-2 sm:row-start-2 sm:row-span-2 lg:mt-6 lg:col-start-1 lg:row-start-3 lg:row-end-4">
        {/* Sử dụng biến loading để kiểm tra trạng thái loading và hiển thị button tương ứng */}
        {loading ? (
          <button
            type="button"
            className="bg-indigo-600 hover:bg-blue-400 text-white text-sm leading-6 font-medium py-2 px-3 rounded-lg"
            disabled
          >
            
            Processing...
          </button>
        ) : (
          <button
            onClick={handleSendData}
            type="button"
            className="bg-indigo-600 hover:bg-blue-400 text-white text-sm leading-6 font-medium py-2 px-3 rounded-lg"
          >
            Generation
          </button>
        )}
      </div>

      {apiResponse && showApiResponse && (
        <div className="mt-2">
          <textarea
            value={apiResponse}
            onChange={(e) => setApiResponse(e.target.value)}
            placeholder="API Response (Lowercase)"
            rows="10"
            className="mt-2 p-2 w-full rounded border border-black rounded-md focus:ring-blue-500 focus:border-blue-500 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
          />
        </div>
      )}
    </div>
  );
};

export default FileUploader;